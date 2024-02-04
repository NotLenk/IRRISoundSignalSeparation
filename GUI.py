import tkinter as tk
from tkinter import filedialog

def choose_file():
    file_path = filedialog.askopenfilename()
    file_path_var.set(file_path)

def run_process():
    # Add your code here to run a process using the selected file path
    print("Running process with file:", file_path_var.get())

# Create the main window
root = tk.Tk()
root.title("Source Separation")

# StringVar to store the file path
file_path_var = tk.StringVar()

# Create the "Choose File" button
choose_file_button = tk.Button(root, text="Choose File", command=choose_file)
choose_file_button.pack(pady=10)

# Create the textbox to display the file path
file_path_entry = tk.Entry(root, textvariable=file_path_var, state="readonly", width=40)
file_path_entry.pack(pady=10)

# Create the "Run" button
run_button = tk.Button(root, text="Run", command=run_process)
run_button.pack()

# Start the GUI event loop
root.mainloop()