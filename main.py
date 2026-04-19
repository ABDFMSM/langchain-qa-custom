import os
from colorama import Fore, init
from query import RAGService
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
init(autoreset=True)

class CLIChatbot:
    def __init__(self):
        print(Fore.YELLOW + "Initializing RAG Service...")
        self.service = RAGService()
        self.chat_history = []

    def start(self):
        instructions = (
            "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
        )
        print(Fore.BLUE + "\x1B[3m" + instructions + "\x1B[0m")

        while True:
            print("\nMENU")
            print("====")
            print("[1]- Ask a question")
            print("[2]- Exit")
            choice = input("Enter your choice: ")
            
            if choice == "1":
                self.ask_loop()
            elif choice == "2":
                print(Fore.GREEN + "Goodbye!")
                break
            else:
                print(Fore.RED + "Invalid choice")

    def ask_loop(self):
        print(Fore.CYAN + "\n--- Q&A Session (type 'x' to return to menu) ---")
        while True:
            user_input = input(Fore.WHITE + "Q: ")
            
            if user_input.lower() == "x":
                break
            
            if not user_input.strip():
                continue

            print(Fore.YELLOW + "Thinking...", end="\r")
            response = self.service.query(user_input, self.chat_history)
            answer = response["answer"]
            
            print(Fore.GREEN + f"A: {answer}")
            
            # Update history
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=answer))
            
            print(Fore.WHITE + "-" * 50)

if __name__ == "__main__":
    try:
        bot = CLIChatbot()
        bot.start()
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
