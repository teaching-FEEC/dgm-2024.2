from colorama import Fore, Style

def print_err(error_message):
       print(f"{Fore.RED}{error_message}{Style.RESET_ALL}")  # Exibe em vermelho no console 
def print_debug(debug_msge):
       print(f"{Fore.GREEN}{debug_msge}{Style.RESET_ALL}")  # Exibe em vermelho no console 