import tkinter as tk
from tkinter import scrolledtext
import threading
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

# Configurazioni RAG
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")

class ChatApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Chatbot")
        self.geometry("800x600")
        
        # Configurazione RAG
        self.embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.llm = Ollama(model="phi4:14b")
        
        # Creazione interfaccia
        self.create_widgets()
        
    def create_widgets(self):
        # Area della chat
        self.chat_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, state='disabled')
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Frame per l'input
        input_frame = tk.Frame(self)
        input_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.user_input = tk.Entry(input_frame, width=70)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", lambda event: self.send_message())
        
        send_button = tk.Button(input_frame, text="Invia", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=(10, 0))
        
    def send_message(self):
        user_text = self.user_input.get()
        if not user_text:
            return
        
        self.display_message("Tu: " + user_text + "\n", "user")
        self.user_input.delete(0, tk.END)
        
        # Esegui la query in un thread separato per evitare il blocco della GUI
        threading.Thread(target=self.process_query, args=(user_text,)).start()
        
    def process_query(self, query_text):
        try:
            # Esegui la query RAG
            response = self.query_rag(query_text)
            self.display_message("Bot: " + response + "\n\n", "bot")
        except Exception as e:
            self.display_message(f"Errore: {str(e)}\n\n", "error")
        
    def query_rag(self, query_text: str):
        # Ricerca nel database vettoriale
        results = self.db.similarity_search_with_score(query_text, k=5)
        
        # Costruzione del contesto
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Generazione della risposta
        response_text = self.llm.invoke(prompt)
        
        # Estrazione delle fonti
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        return f"{response_text}\nFonti: {sources}\n"
        
    def display_message(self, message, sender):
        self.chat_area.configure(state='normal')
        tag = sender
        self.chat_area.insert(tk.END, message, tag)
        
        # Configura formattazione
        self.chat_area.tag_config("user", foreground="blue")
        self.chat_area.tag_config("bot", foreground="green")
        self.chat_area.tag_config("error", foreground="red")
        
        self.chat_area.configure(state='disabled')
        self.chat_area.see(tk.END)

if __name__ == "__main__":
    app = ChatApplication()
    app.mainloop()