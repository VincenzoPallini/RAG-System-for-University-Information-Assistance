import os
import sys
import json
import re
import numpy as np
import logging
from tqdm import tqdm
from langchain_ollama import OllamaLLM  
from langchain_ollama import OllamaEmbeddings  
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from ragelo import get_retrieval_evaluator  
import asyncio
import nest_asyncio
import ast

# Forza l'uso di UTF-8 per stdout (utile su Windows per evitare UnicodeEncodeError)
sys.stdout.reconfigure(encoding='utf-8')

nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = "dummy"

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_json_string(s: str) -> str:
    """Rimuove caratteri di controllo non validi, esegue l'escape dei newline,
    converte virgolette tipografiche in virgolette standard e rimuove eventuali virgole in eccesso.
    """
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', s)
    s = re.sub(r'(?<!\\)\n', '\\n', s)
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r'(?<=\w)"(?=\w)', r'\\"', s)
    s = re.sub(r',\s*([\]}])', r'\1', s)
    return s

def clean_response(response):
    """Rimuove delimitatori di blocco (es. ```json) dalla risposta."""
    response = response.strip()
    if response.startswith("```"):
        lines = response.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines)
    return response

def extract_json(text):
    """Estrae il primo blocco JSON trovato (o restituisce il testo originale)."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text

def normalize_ground_truth(gt):
    """Normalizza il dizionario della ground truth:
    - Rinomina 'alternatives_answers' in 'alternative_answers'
    - Rimuove eventuali stringhe vuote dalla lista.
    """
    if "alternatives_answers" in gt:
        gt["alternative_answers"] = gt.pop("alternatives_answers")
    if "alternative_answers" in gt and isinstance(gt["alternative_answers"], list):
        gt["alternative_answers"] = [ans for ans in gt["alternative_answers"] if ans.strip()]
    return gt

class CustomCRAG:
    def __init__(self, pdf_folder="data"):
        self.pdf_folder = pdf_folder
        self.qa_pairs = []
        self.vector_db = None
        # Model used to generate questions and ground truth
        self.llm = OllamaLLM(model="gemma2:2b")
        # Separate model used for evaluation
        self.eval_llm = OllamaLLM(model="phi4:14b")
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize_ragelo_evaluator()
        self.initialize_system()
        
    def initialize_ragelo_evaluator(self):
        """
        Inizializza l'evaluator di RAGElo con un prompt che richiede la valutazione in formato JSON.
        Il prompt chiede all'LLM di restituire le chiavi "quality", "trustworthiness" e "originality".
        """
        custom_prompt = (
            "Valuta la seguente risposta generata rispetto alla risposta attesa. "
            "Assegna un punteggio numerico da 0 a 1 per ciascuna delle seguenti dimensioni, in base alla pertinenza e alla qualità effettiva della risposta:\n"
            "- quality: qualità complessiva e pertinenza della risposta\n"
            "- trustworthiness: accuratezza e affidabilità delle informazioni fornite\n"
            "- originality: chiarezza e originalità della risposta\n\n"
            "Domanda: {q}\n"
            "Risposta generata e attesa: {d}\n\n"
            "Rispondi SOLO con un oggetto JSON che segua esattamente questo formato, sostituendo i segnaposto con i valori dinamici risultanti dalla tua valutazione:\n"
            "{{\"quality\": <valore>, \"trustworthiness\": <valore>, \"originality\": <valore>}}"
        )
    
        response_schema = {
            "quality": "Un numero tra 0 e 1 che rappresenta la qualità complessiva e la pertinenza della risposta",
            "trustworthiness": "Un numero tra 0 e 1 che rappresenta l'accuratezza e l'affidabilità delle informazioni fornite",
            "originality": "Un numero tra 0 e 1 che rappresenta la chiarezza e l'originalità della risposta"
        }
        # Utilizzo il modello di valutazione definito in self.eval_llm
        self.ragelo_evaluator = get_retrieval_evaluator(
            "custom_prompt",
            llm_provider="ollama",
            prompt=custom_prompt,
            query_placeholder="q",
            document_placeholder="d",
            llm_answer_format="json",
            llm_response_schema=response_schema,
            seed=42,
            model=self.eval_llm.model,
            openai_api_key="dummy"
        )
    
    def initialize_system(self):
        """Carica e indicizza i PDF e crea il vector store."""
        documents = []
        logging.info("=== Caricamento PDF ===")
        if not os.path.exists(self.pdf_folder):
            logging.error(f"La cartella '{self.pdf_folder}' non esiste.")
            return
        for file in os.listdir(self.pdf_folder):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, file)
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                logging.info(f"{file}: {len(docs)} documenti caricati")
                documents.extend(docs)
        logging.info(f"Totale documenti caricati: {len(documents)}")
        if not documents:
            logging.error("Nessun documento PDF caricato. Interrompo l'inizializzazione.")
            return
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Numero di chunk generati: {len(chunks)}")
        if not chunks:
            logging.error("Nessun chunk generato. Interrompo l'inizializzazione.")
            return
        self.vector_db = Chroma.from_documents(
            chunks, embedding=OllamaEmbeddings(model="nomic-embed-text"), persist_directory="chroma_db"
        )
        vector_docs = self.vector_db.get().get('documents', [])
        logging.info(f"Documenti indicizzati nel vector store: {len(vector_docs)}")
        self.generate_test_dataset()
    
    def generate_test_dataset(self):
        """Crea QA pairs dai documenti seguendo lo schema CRAG."""
        vector_docs = self.vector_db.get().get('documents', [])
        if not vector_docs:
            logging.error("Nessun documento trovato nel vector store per generare QA pairs.")
            return
        logging.info("=== Generazione QA pairs ===")
        max_chunks = 100
        count = 0
        for chunk in tqdm(vector_docs, desc="Generazione QA pairs"):
            if count >= max_chunks:
                break
            if isinstance(chunk, str):
                chunk_text = chunk
            else:
                chunk_text = getattr(chunk, "page_content", str(chunk))
            questions = self.generate_questions(chunk_text, num_questions=1)
            logging.info(f"Domande generate: {questions}")
            for q in questions:
                gt = self.generate_ground_truth(q, chunk_text)
                if not gt.get("answer"):
                    logging.warning(f"Ground truth per la domanda '{q}' vuota, saltando questo QA pair.")
                    continue
                logging.info(f"Per la domanda '{q}' è stata generata la ground truth: {gt}")
                self.qa_pairs.append({
                    "question": q,
                    "ground_truth": gt,
                    "domain": "custom",
                    "question_type": "factual",
                    "context": chunk_text
                })
            count += 1
        logging.info(f"Totale QA pairs generati: {len(self.qa_pairs)}")
    
    def generate_questions(self, context, num_questions=1):
        """Genera domande dal contesto usando la metodologia CRAG."""
        prompt = f"""
Genera 2-3 domande che questo contesto può rispondere seguendo lo schema CRAG:
{context[:1500]}...

Formato richiesto (JSON):
{{
    "questions": ["domanda1", "domanda2"],
    "question_types": ["factual", "comparison"]
}}
        """
        response = self.llm.invoke(prompt)
        response_clean = clean_response(response)
        response_clean = extract_json(response_clean)
        if not response_clean.strip().endswith("}"):
            response_clean = self.balance_json(response_clean)
        try:
            data = json.loads(sanitize_json_string(response_clean), strict=False)
            questions = data.get('questions', [])
            return questions[:num_questions]
        except Exception as e:
            logging.error(f"Errore nella generazione delle domande: {e}. Risposta non valida: {response_clean}")
            return []
    
    def generate_ground_truth(self, question, context):
        """Genera una risposta precisa dal contesto."""
        prompt = f"""
Genera una risposta precisa per la domanda basandoti ESCLUSIVAMENTE sul contesto.
Domanda: {question}
Contesto: {context}

Formato richiesto:
{{
    "answer": "risposta precisa",
    "alternative_answers": ["alt1", "alt2"]
}}
        """
        response = self.llm.invoke(prompt)
        response_clean = clean_response(response)
        response_clean = extract_json(response_clean)
        if not response_clean.strip().endswith("}"):
            response_clean = response_clean.strip() + "}"
        response_clean = self.balance_json(response_clean)
        try:
            gt = json.loads(sanitize_json_string(response_clean), strict=False)
            gt = normalize_ground_truth(gt)
            return gt
        except Exception as e:
            logging.error(f"Errore nella generazione della ground truth per la domanda '{question}': {e}. Risposta non valida: {response_clean}")
            return {"answer": "", "alternative_answers": []}
    
    def balance_json(self, s):
        """
        Verifica e bilancia le parentesi graffe e quadre nel JSON.
        Se mancano chiusure, le aggiunge alla fine.
        """
        open_braces = s.count("{")
        close_braces = s.count("}")
        if open_braces > close_braces:
            s += "}" * (open_braces - close_braces)
        open_brackets = s.count("[")
        close_brackets = s.count("]")
        if open_brackets > close_brackets:
            s += "]" * (open_brackets - close_brackets)
        return s
    
    def llm_evaluation_judge(self, question, generated_answer, ground_truth, qid="<no_qid>", did="<no_did>"):
        """
        Valuta le risposte dell'LLM mappando le metriche ottenute con quelle attese.
        """
        logger = logging.getLogger(__name__)
        
        expected_answer = ground_truth.get("answer", "")
        document_text = f"Risposta generata: {generated_answer}\nRisposta attesa: {expected_answer}"
        
        def clean_and_parse_json(json_str):
            try:
                json_str = json_str.strip()
                json_str = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.DOTALL)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'\s*:\s*', ':', json_str)
                json_str = re.sub(r'\s*,\s*', ',', json_str)
                json_str = re.sub(r'([{,])(\s*[^"\s][^:}]*?):', r'\1"\2":', json_str)
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                return json.loads(json_str)
            except Exception as e:
                logger.error(f"Error in clean_and_parse_json for qid: {qid}, did: {did}: {str(e)}", exc_info=True)
                return {"relevance": 0.0, "accuracy": 0.0, "clarity": 0.0}
        
        try:
            try:
                result = self.ragelo_evaluator.evaluate(query=question, document=document_text)
            except RuntimeError as err:
                if "no running event loop" in str(err):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(self.ragelo_evaluator._async_evaluate((question, document_text)))
                    finally:
                        loop.close()
                else:
                    raise
            if isinstance(result, tuple):
                result_data = result[0]
            else:
                result_data = result

            if isinstance(result_data, str):
                result_data = result_data.strip()
                json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', result_data, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    result_data = json.loads(json_str)
                else:
                    result_data = json.loads(result_data)
            
            mapped_metrics = {
                "quality": float(result_data.get("quality", 0.0)),
                "trustworthiness": float(result_data.get("trustworthiness", 0.0)),
                "originality": float(result_data.get("originality", 0.0))
            }
            
            for key in ["quality", "trustworthiness", "originality"]:
                if key not in mapped_metrics:
                    logger.warning(f"Missing key {key} for qid {qid}. Using default value 0.0.")
                    mapped_metrics[key] = 0.0
            
            overall = sum(mapped_metrics.values()) / len(mapped_metrics)
            return min(max(overall, 0.0), 1.0)
        
        except Exception as e:
            logger.error(f"Evaluation failed for qid: {qid}, did: {did}: {str(e)}", exc_info=True)
            logger.debug(f"Question: {question}")
            logger.debug(f"Generated answer: {generated_answer}")
            logger.debug(f"Ground truth: {ground_truth}")
            return 0.0
    
    def refine_answer(self, question, current_answer, context, ground_truth):
        logging.info(f"Raffinamento risposta per la domanda: {question}")
        additional_docs = self.vector_db.similarity_search(f"{question} {current_answer}", k=3)
        additional_context = "\n".join([getattr(doc, "page_content", str(doc)) for doc in additional_docs])
        combined_context = context + "\n" + additional_context
        prompt = f"""
La risposta seguente alla domanda potrebbe essere migliorata integrando ulteriori dettagli dal contesto:
Domanda: {question}
Risposta attuale: {current_answer}
Contesto esteso: {combined_context}

Raffina la risposta in modo che risulti più accurata e completa, mantenendo il formato JSON come:
{{"answer": "risposta raffinata", "alternative_answers": ["alt1", "alt2"]}}
        """
        refined_response = self.llm.invoke(prompt)
        refined_clean = clean_response(refined_response)
        refined_clean = extract_json(refined_clean)
        try:
            refined_answer = json.loads(sanitize_json_string(refined_clean), strict=False)
            refined_answer = normalize_ground_truth(refined_answer)
            logging.info(f"Risposta raffinata per la domanda '{question}': {refined_answer}")
            return refined_answer
        except Exception as e:
            logging.error(f"Errore nel raffinamento della risposta per la domanda '{question}': {e}. Risposta: {refined_clean}")
            return {"answer": current_answer, "alternative_answers": []}
    
    def evaluate_rag(self):
        results = []
        if not self.qa_pairs:
            logging.error("Nessun QA pair generato. Impossibile eseguire la valutazione.")
            return self.analyze_results(results)
        logging.info("=== Inizio valutazione CRAG Personalizzata ===")
        for qa in tqdm(self.qa_pairs, desc="Valutazione CRAG Personalizzata"):
            retrieved_docs = self.vector_db.similarity_search(qa["question"], k=5)
            context = "\n".join([getattr(doc, "page_content", str(doc)) for doc in retrieved_docs])
            prompt = f"Domanda: {qa['question']}\nContesto: {context}\nRisposta:"
            generated_answer = self.llm.invoke(prompt)
            # Valutazione diretta della risposta generata senza raffinamento
            evaluation = self.crag_evaluation(
                question=qa["question"],
                prediction=generated_answer,
                ground_truth=qa["ground_truth"],
                retrieved_context=context
            )
            evaluation["prediction"] = generated_answer
            evaluation["ground_truth"] = qa["ground_truth"]
            evaluation["question"] = qa["question"]
            evaluation["retrieved_docs"] = [getattr(doc, "page_content", str(doc))[:200] for doc in retrieved_docs]
            results.append(evaluation)
        return self.analyze_results(results)
    
    def crag_evaluation(self, question, prediction, ground_truth, retrieved_context):
        if isinstance(prediction, dict):
            prediction_text = prediction.get("answer", "")
        else:
            prediction_text = prediction
        if "non lo so" in prediction_text.lower():
            base_score = 0
            category = "missing"
        else:
            exact_match = any(
                prediction_text.lower().strip() == ans.lower().strip() 
                for ans in [ground_truth.get("answer", "")] + ground_truth.get("alternative_answers", [])
            )
            if exact_match:
                base_score = 1
                category = "exact_match"
            else:
                gt_emb = self.sim_model.encode(ground_truth.get("answer", ""))
                pred_emb = self.sim_model.encode(prediction_text)
                similarity = np.dot(gt_emb, pred_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(pred_emb) + 1e-8)
                if similarity > 0.8:
                    base_score = 1
                    category = "semantic_match"
                elif similarity > 0.5:
                    base_score = 0.5
                    category = "partial_match"
                else:
                    base_score = 0
                    category = "hallucination"
        
        try:
            result = self.ragelo_evaluator.evaluate(
                query=question,
                document=f"Risposta generata: {prediction_text}\nRisposta attesa: {ground_truth.get('answer', '')}"
            )
        except RuntimeError as err:
            if "no running event loop" in str(err):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.ragelo_evaluator._async_evaluate(
                        (question, f"Risposta generata: {prediction_text}\nRisposta attesa: {ground_truth.get('answer', '')}")
                    ))
                finally:
                    loop.close()
            else:
                raise
        if isinstance(result, tuple):
            result_data = result[0]
        else:
            result_data = result

        if isinstance(result_data, str):
            result_data = result_data.strip()
            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', result_data, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                result_data = json.loads(json_str)
            else:
                result_data = json.loads(result_data)
        
        mapped_metrics = {
            "quality": float(result_data.get("quality", 0.0)),
            "trustworthiness": float(result_data.get("trustworthiness", 0.0)),
            "originality": float(result_data.get("originality", 0.0))
        }
        
        for key in ["quality", "trustworthiness", "originality"]:
            if key not in mapped_metrics:
                logging.warning(f"Missing key {key} for question {question}. Using default value 0.0.")
                mapped_metrics[key] = 0.0
        
        overall_llm_score = sum(mapped_metrics.values()) / len(mapped_metrics)
        overall = (base_score + overall_llm_score) / 2
        return {
            "score": min(max(overall, 0.0), 1.0),
            "category": category,
            "fact_score": base_score,
            "llm_score": overall_llm_score,
            "quality": mapped_metrics["quality"],
            "trustworthiness": mapped_metrics["trustworthiness"],
            "originality": mapped_metrics["originality"]
        }
    
    def analyze_results(self, results):
        if not results:
            logging.error("Nessun risultato da analizzare. Verifica il processo di generazione dei QA pairs.")
            return {
                "overall_score": 0,
                "accuracy": 0,
                "hallucination_rate": 0,
                "coverage": 0,
                "error_analysis": {"retrieval_errors": [], "generation_errors": []}
            }
        scores = [r["score"] for r in results]
        categories = [r["category"] for r in results]
        overall_score = np.mean(scores)
        accuracy = sum(1 for s in scores if s > 0.5) / len(scores)
        hallucination_rate = sum(1 for c in categories if c == "hallucination") / len(categories)
        coverage = 1 - (sum(1 for c in categories if c == "missing") / len(categories))
        return {
            "overall_score": overall_score,
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "coverage": coverage,
            "error_analysis": {
                "retrieval_errors": self.analyze_retrieval(results),
                "generation_errors": self.analyze_generation(results)
            }
        }
    
    def analyze_retrieval(self, results):
        error_samples = []
        for res in results:
            if res["score"] < 0 and res["category"] != "missing":
                error_samples.append({
                    "question": res["question"],
                    "retrieved_context": res["retrieved_docs"]
                })
        return error_samples[:5]
    
    def analyze_generation(self, results):
        errors = []
        for res in results:
            if res["score"] < 0 and res["category"] != "missing":
                errors.append({
                    "question": res["question"],
                    "predicted_answer": res.get("prediction", ""),
                    "expected_answer": res.get("ground_truth", {}).get("answer", "")
                })
        return errors[:5]

if __name__ == "__main__":
    evaluator = CustomCRAG(pdf_folder="data")
    
    # Esegui la valutazione aggregata
    aggregated_results = evaluator.evaluate_rag()
    logging.info("\n" + "="*50)
    logging.info("🔬 Valutazione CRAG Personalizzata sui Tuoi Dati 🔬")
    logging.info(f"Punteggio Complessivo: {aggregated_results.get('overall_score', 0):.2f}/1.0")
    logging.info(f"Accuratezza: {aggregated_results.get('accuracy', 0):.2%}")
    logging.info(f"Tasso Allucinazioni: {aggregated_results.get('hallucination_rate', 0):.2%}")
    logging.info(f"Copertura: {aggregated_results.get('coverage', 0):.2%}")
    
    logging.info("\n🚨 Errori di Retrieval Campione:")
    for error in aggregated_results.get('error_analysis', {}).get('retrieval_errors', []):
        logging.info(f"Domanda: {error['question']}")
        for ctx in error['retrieved_context']:
            logging.info(f"- {ctx}...")
    
    logging.info("\n🚨 Errori di Generazione Campione:")
    for error in aggregated_results.get('error_analysis', {}).get('generation_errors', []):
        logging.info(f"Domanda: {error['question']}")
        logging.info(f"Risposta Generata: {error.get('predicted_answer', '')}")
        logging.info(f"Risposta Attesa: {error.get('ground_truth', {}).get('answer', '')}")
    
    # Costruisci l'elenco dettagliato delle valutazioni per ogni QA pair
    detailed_evaluations = []
    for qa in evaluator.qa_pairs:
        retrieved_docs = evaluator.vector_db.similarity_search(qa["question"], k=5)
        context = "\n".join([getattr(doc, "page_content", str(doc)) for doc in retrieved_docs])
        prompt = f"Domanda: {qa['question']}\nContesto: {context}\nRisposta:"
        generated_answer = evaluator.llm.invoke(prompt)
        evaluation = evaluator.crag_evaluation(
            question=qa["question"],
            prediction=generated_answer,
            ground_truth=qa["ground_truth"],
            retrieved_context=context
        )
        evaluation["prediction"] = generated_answer
        evaluation["ground_truth"] = qa["ground_truth"]
        evaluation["question"] = qa["question"]
        evaluation["retrieved_docs"] = [getattr(doc, "page_content", str(doc))[:200] for doc in retrieved_docs]
        detailed_evaluations.append(evaluation)
    
    # Logga l'output dettagliato in un file
    output_lines = []
    output_lines.append("="*50)
    output_lines.append("🔬 Valutazione CRAG Personalizzata sui Tuoi Dati 🔬")
    output_lines.append(f"Punteggio Complessivo: {aggregated_results.get('overall_score', 0):.2f}/1.0")
    output_lines.append(f"Accuratezza: {aggregated_results.get('accuracy', 0):.2%}")
    output_lines.append(f"Tasso Allucinazioni: {aggregated_results.get('hallucination_rate', 0):.2%}")
    output_lines.append(f"Copertura: {aggregated_results.get('coverage', 0):.2%}")
    output_lines.append("\nDettaglio per ogni QA pair:")
    
    for qa in detailed_evaluations:
        output_lines.append("-"*40)
        output_lines.append(f"Domanda: {qa.get('question', '')}")
        gt = qa.get("ground_truth", {})
        output_lines.append(f"Ground Truth: {gt.get('answer', '')} (Alternative: {gt.get('alternative_answers', [])})")
        output_lines.append(f"Risposta Generata: {qa.get('prediction', '')}")
        output_lines.append(f"Quality: {qa.get('quality', 0):.2f}")
        output_lines.append(f"Trustworthiness: {qa.get('trustworthiness', 0):.2f}")
        output_lines.append(f"Originality: {qa.get('originality', 0):.2f}")
        output_lines.append(f"Fact Score: {qa.get('fact_score', 0):.2f}")
        output_lines.append(f"LLM Score: {qa.get('llm_score', 0):.2f}")
    
    output_text = "\n".join(output_lines)
    with open("evaluation_output.txt", "w", encoding="utf-8") as file:
        file.write(output_text)
    logging.info("Output salvato in evaluation_output.txt")
