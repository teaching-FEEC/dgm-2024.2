import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader, PdfWriter

# Função para salvar figuras em um novo PDF ou adicionar em um existente
def save_fig_pdf(file, *figs, append=False):
    if append:
        if os.path.exists(file):
            # Copiar o conteúdo existente para um novo arquivo PDF
            writer = PdfWriter()
            reader = PdfReader(file)
            for page in reader.pages:
                writer.add_page(page)
            
            # Criar um arquivo temporário para salvar o PDF com as novas figuras
            temp_pdf = 'temp_append.pdf'
            with PdfPages(temp_pdf) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
                    plt.close(fig)
            
            # Combinar o arquivo original com o temporário
            with open(file, 'wb') as final_pdf, open(temp_pdf, 'rb') as append_pdf:
                writer.append(append_pdf)
                writer.write(final_pdf)
            
            # Remover o arquivo temporário
            os.remove(temp_pdf)
        else:
            # Arquivo não existe, criar um novo e salvar as figuras
            with PdfPages(file) as pdf:
                for fig in figs:
                    pdf.savefig(fig)
                    plt.close(fig)
    else:
        # Criar um novo arquivo PDF e salvar as figuras
        with PdfPages(file) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Figuras salvas no arquivo '{file}'.")
    