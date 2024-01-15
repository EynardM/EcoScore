# Import necessary modules and functions
from colorama import Fore, Style
import inspect
import os

# Function to print a variable name and its value in colored text
def print_colored(variable, color):
    # Define color codes using colorama
    color_map = {
        "blue": Fore.BLUE,
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "cyan": Fore.CYAN,
        "magenta": Fore.MAGENTA,
        "white": Fore.WHITE,
        "bright_black": Fore.BLACK + Style.BRIGHT,
        "bright_red": Fore.RED + Style.BRIGHT,
        "bright_green": Fore.GREEN + Style.BRIGHT,
        "bright_yellow": Fore.YELLOW + Style.BRIGHT,
        "bright_cyan": Fore.CYAN + Style.BRIGHT,
        "bright_magenta": Fore.MAGENTA + Style.BRIGHT,
        "bright_white": Fore.WHITE + Style.BRIGHT,
    }

    # Check if the specified color is supported
    if color not in color_map:
        print("Unsupported color.")
        return

    # Extract color codes for the specified color
    color_code = color_map[color]
    reset_code = Style.RESET_ALL

    # Get the calling frame to determine the variable name
    frame = inspect.currentframe().f_back
    variable_name = [name for name, value in frame.f_locals.items() if value is variable][0]

    # Print the variable name and value in colored text
    print(f"{color_code}{variable_name} = {variable}{reset_code}")
