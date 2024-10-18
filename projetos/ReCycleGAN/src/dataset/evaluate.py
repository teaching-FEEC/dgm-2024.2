from enum import StrEnum
import os
import pandas as pd
from pathlib import Path
import sys
import tempfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.evaluation import Result
from ragas.metrics.critique import harmfulness
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator
from tqdm.contrib.concurrent import thread_map
import wandb

from app.core.bot import Bot, Reply
from app.tdxUtils.tdx_text_utils import DeveloperDocumentLoader
from scripts.utils import Constants

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["API_KEY_OPENAI"]

DatasetType = StrEnum("DatasetType", ["synthetic", "organic"])


def eval(
    bot: Bot,
    experiment_name: str,
    experiment_description: str,
    ds_type: DatasetType = DatasetType.organic,
    synthetic_size: int = 100,
    synthetic_dist: dict = {"simple": 0.4, "reasoning": 0.3, "multi_context": 0.3},
    synthetic_generator: str = "gpt-4o-mini",
    synthetic_critic: str = "gpt-4o",
    synthetic_embeddings: str = "text-embedding-3-small",
    eval_llm: str = "gpt-4o",
    eval_embeddings: str = "text-embedding-3-small",
):

    config = {
        "experiment_name": experiment_name,
        "experiment_description": experiment_description,
        "ds_type": ds_type,
        "synthetic_size": synthetic_size,
        "synthetic_dist": synthetic_dist,
        "synthetic_generator": synthetic_generator,
        "synthetic_critic": synthetic_critic,
        "synthetic_embeddings": synthetic_embeddings,
        "eval_llm": eval_llm,
        "eval_embeddings": eval_embeddings,
    }

    wandb_run = wandb.init(
        project=Constants.WB_PROJECT,
        name=config["experiment_name"],
        notes=config["experiment_description"],
        job_type=Constants.WB_BOT_EVAL_JOB,
        config=config,
    )

    with wandb_run:
        eval_db = _load_eval_db(wandb_run, config)
        replies = _get_replies(bot, eval_db)
        result = _evaluate_replies(eval_db, replies, config)

        for metric, value in result.items():
            wandb_run.summary[metric] = value

        result_df = result.to_pandas()
        result_df["contexts"] = result_df["contexts"].map(
            lambda contexts: "".join(
                [f"# Context {i + 1}:\n{c}" for i, c in enumerate(contexts)]
            )
        )

        _log_df_as_table(
            input_df=result_df[[m for m in result]].describe(),
            title="Performance Summary",
            wandb_run=wandb_run,
        )
        wandb_run.log({"performance_table": wandb.Table(dataframe=result_df)})


def _load_knowledge_base():
    try:
        kb_art_name = Path(Constants.KB_LOCAL_FILEPATH).stem
        api = wandb.Api()
        kb_art = api.artifact(f"{Constants.WB_PROJECT}/{kb_art_name}:latest")
    except:
        print(
            f"Artifact '{kb_art_name}' does not exist."
            "Make sure to push the knowledge base to WandB first."
        )
        exit()

    with tempfile.TemporaryDirectory() as temp_dir:
        art_local_dir = Path(temp_dir)
        kb_art.download(root=art_local_dir)
        loader = DeveloperDocumentLoader(art_local_dir / Constants.WB_KB_NAME)
        documents = [doc for doc in loader.load()]

    return documents, kb_art.version


def _load_eval_db(wandb_run, config: dict):
    if config["ds_type"] == DatasetType.synthetic:
        return _load_syn_eval_db(wandb_run, config)
    else:
        return _load_org_eval_db(wandb_run)


def _load_org_eval_db(wandb_run):
    try:
        org_art_name = Path(Constants.ORGANIC_EVAL_LOCAL_FILEPATH).stem
        org_art = wandb_run.use_artifact(f"{org_art_name}:latest")
    except:
        print(
            f"Artifact '{org_art_name}' does not exist."
            "Make sure to push the organic eval database to WandB first."
        )
        exit()

    print(f"Organic eval DB cache loaded from WandB ('{org_art_name}').")
    wandb_run.tags = wandb_run.tags + (f"ODB_{org_art.version}",)

    with tempfile.TemporaryDirectory() as temp_dir:
        org_eval_db_local_dir = Path(temp_dir)
        org_art.download(root=org_eval_db_local_dir)
        org_eval_db_filename = Path(Constants.ORGANIC_EVAL_LOCAL_FILEPATH).name
        org_db_local_filepath = org_eval_db_local_dir / org_eval_db_filename
        df = pd.read_csv(org_db_local_filepath)
        ragas_eval_df = df.rename(
            columns={"question_raw": "question", "solution_raw": "ground_truth"},
            errors="raise",
        )[["question", "ground_truth"]]
        return ragas_eval_df


def _load_syn_eval_db(wandb_run, config: dict):

    try:
        kb_art_name = Path(Constants.KB_LOCAL_FILEPATH).stem
        kb_art = wandb_run.use_artifact(f"{kb_art_name}:latest")
    except:
        print(
            f"Artifact '{kb_art_name}' does not exist."
            "Make sure to push the knowledge base to WandB first."
        )
        exit()

    wandb_run.tags = wandb_run.tags + (f"KB_{kb_art.version}",)

    try:
        syn_eval_art_name = f"{Constants.WB_SYN_EVAL_NAME}:latest"
        syn_eval_art = wandb_run.use_artifact(syn_eval_art_name)
        if syn_eval_art.metadata.get("kb_version", "") != kb_art.version:
            raise ValueError(f"Cached syn db not compatible with KB {art.version}.")
        print(f"Synthetic eval DB cache loaded from WandB ('{syn_eval_art_name}').")

        with tempfile.TemporaryDirectory() as temp_dir:
            syn_eval_db_local_dir = Path(temp_dir)
            syn_eval_art.download(root=syn_eval_db_local_dir)
            return pd.read_parquet(syn_eval_db_local_dir)
    except:
        print(f"Generating a new synthetic eval DB as '{syn_eval_art_name}'...")
        testset_df = _generate_syn_eval_db(kb_art, config)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".parquet.gzip", delete=False
        ) as temp_file:
            file_path = temp_file.name
            testset_df.to_parquet(file_path)
            art = wandb.Artifact(
                name=Constants.WB_SYN_EVAL_NAME,
                type=Constants.WB_DB_ARTIFACT_TYPE,
                metadata={"kb_version": kb_art.version},
            )
            art.add_file(local_path=file_path)
            wandb_run.log_artifact(artifact_or_path=art)
        return testset_df


def _generate_syn_eval_db(kb_art, config: dict) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as temp_dir:
        art_local_dir = Path(temp_dir)
        kb_art.download(root=art_local_dir)
        loader = DeveloperDocumentLoader(art_local_dir / Constants.WB_KB_NAME)
        documents = [doc for doc in loader.load()]

    for doc in documents:
        doc.metadata["filename"] = doc.metadata["url"]

    generator_llm = ChatOpenAI(model=config["synthetic_generator"], temperature=0)
    critic_llm = ChatOpenAI(model=config["synthetic_critic"], temperature=0)
    embeddings = OpenAIEmbeddings(model=config["synthetic_embeddings"])
    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

    synthetic_dist = config["synthetic_dist"]
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=config["synthetic_size"],
        distributions={
            simple: synthetic_dist["simple"],
            reasoning: synthetic_dist["reasoning"],
            multi_context: synthetic_dist["multi_context"],
        },
    )
    return testset.to_pandas()


def _get_replies(bot: Bot, eval_db: pd.DataFrame) -> list[Reply]:
    return thread_map(
        lambda query: bot.reply(query),
        eval_db["question"].array,
        desc="Running bot inferences",
    )
    # import time
    # from tqdm import tqdm

    # reps = []
    # for q in tqdm(eval_db["question"].array, desc="Getting replies"):
    #     reps.append(bot.reply(q))
    #     # time.sleep(10)
    # return reps


def _evaluate_replies(
    eval_db: pd.DataFrame,
    replies: list[Reply],
    config: dict,
) -> Result:
    data_dict = {
        "question": eval_db["question"],
        "ground_truth": eval_db["ground_truth"],
        "answer": [r.answer for r in replies],
        "contexts": [r.contexts for r in replies],
    }
    scores = evaluate(
        dataset=Dataset.from_dict(data_dict),
        metrics=[
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
            harmfulness,
        ],
        llm=ChatOpenAI(model=config["eval_llm"], temperature=0),
        embeddings=OpenAIEmbeddings(model=config["eval_embeddings"]),
    )
    return scores


def _log_df_as_table(input_df: pd.DataFrame, title: str, wandb_run):
    html = _html_from_dataframe(input_df)
    wandb_run.log({title: wandb.Html(html)})


def _html_from_dataframe(df: pd.DataFrame):
    html = df.to_html(classes="table table-striped table-hover", justify="center")
    style = """
<style>
    .table {
        width: 100%;
        border-collapse: collapse;
    }
    .table, .table th, .table td {
        border: 1px solid black;
    }
    th, td {
        padding: 10px;
        text-align: center;
    }
</style>
"""
    return style + html


if __name__ == "__main__":
    from app.core.retrievers.graph_retriever import GraphRetriever
    from app.core.retrievers.naive_vector_retriever import NaiveVectorRetriever
    from app.core.retrievers.parent_vector_retriever import ParentVectorRetriever
    from app.core.retrievers.reranking_vector_retriever import RerankingVectorRetriever

    kb_docs, kb_version = _load_knowledge_base()

    # rag = NaiveVectorRetriever(kb_docs, kb_version)
    # bot = Bot(rag)
    # eval(
    #     bot,
    #     experiment_name="1st round of tests",
    #     experiment_description="Naive vector retriever",
    #     ds_type=DatasetType.synthetic,
    # )
    # eval(
    #     bot,
    #     experiment_name="1st round of tests",
    #     experiment_description="Naive vector retriever",
    #     ds_type=DatasetType.organic,
    # )
    # rag = ParentVectorRetriever(kb_docs, kb_version)
    # bot = Bot(rag)
    # eval(
    #     bot,
    #     experiment_name="1st round of tests",
    #     experiment_description="Parent vector retriever",
    #     ds_type=DatasetType.synthetic,
    # )
    # eval(
    #     bot,
    #     experiment_name="1st round of tests",
    #     experiment_description="Parent vector retriever",
    #     ds_type=DatasetType.organic,
    # )
    # rag = RerankingVectorRetriever(kb_version)
    # bot = Bot(rag)
    # eval(
    #     bot,
    #     experiment_name="1st round of tests",
    #     experiment_description="Reranking vector retriever",
    #     experiment_description="Reranking vector retriever",
    #     ds_type=DatasetType.synthetic,
    # )
    # eval(
    #     bot,
    #     experiment_name="1st round of tests",
    #     experiment_description="Reranking vector retriever",
    #     experiment_description="Reranking vector retriever",
    #     ds_type=DatasetType.organic,
    # )
    rag = GraphRetriever(kb_docs)
    bot = Bot(rag)
    eval(
        bot,
        experiment_name="GraphRAG improvements",
        experiment_description="Microsoft global graph retriever with partial responses",
        ds_type=DatasetType.synthetic,
    )
    eval(
        bot,
        experiment_name="GraphRAG improvements",
        experiment_description="Microsoft global graph retriever with partial responses",
        ds_type=DatasetType.organic,
    )
