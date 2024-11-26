while getopts t:s:c:r:l: flag 
do
        case "${flag}" in
                t) txt=${OPTARG};;
                s) spkrID=${OPTARG};;
                c) audioID=${OPTARG};;
                r) case "${OPTARG}" in
                        "run_01") run="run_01-November-14-2024_08+56PM-0000000";;
                        "run_02") run="run_02-November-15-2024_08+17PM-0000000";;
                        "run_03") run="run_03-November-23-2024_03+57AM-0000000";;
                        "run_04") run="run_04-November-23-2024_08+15PM-0000000";;
                        *) run=${OPTARG};; 
                   esac;;
                l) langID=${OPTARG};
        esac
done

echo "Generating sample $audioID with text $txt in the voice of $spkrID using model from $run";

tts --text "$txt" --speaker_idx "$spkrID" --language_idx "$langID" --model_path /workspace/outputs/$run/best_model.pth --config_path /workspace/outputs/$run/config.json --out_path /workspace/outputs/$run/samples/$spkrID/Synthesized/$audioID.wav