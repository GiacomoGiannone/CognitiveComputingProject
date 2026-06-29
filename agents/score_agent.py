# agents/score_agent.py
"""
Score Agent basato su BERT italiano (dbmdz/bert-base-italian-uncased)
Valuta la qualità di un post con un punteggio da 0 a 1.
Fine-tuning per regressione su dataset di post etichettati.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, TrainerCallback
from langsmith import traceable
from typing import List, Dict
import json
import os
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime


class LossLoggingCallback(TrainerCallback):
    """
    Callback personalizzato per il Trainer di HuggingFace.
    Il Trainer non fornisce di default un grafico della loss: questo callback
    intercetta i log ad ogni step (on_log) e accumula i valori di training loss
    e validation loss per poi generare un grafico a fine training.
    """
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
        self.train_epochs = []
        self.eval_epochs = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
                self.train_epochs.append(state.epoch)
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
                self.eval_epochs.append(state.epoch)
    
    def plot_losses(self, output_dir: str = None):
        """Genera e salva il grafico delle loss"""
        if not self.train_losses:
            print(" Nessuna loss registrata durante il training.")
            return
        
        plt.figure(figsize=(14, 7))
        
        # Plot training loss
        if self.train_losses:
            plt.plot(self.steps, self.train_losses, 
                    label='Training Loss', color='blue', linewidth=2, alpha=0.8)
        
        # Plot evaluation loss
        if self.eval_losses:
            plt.plot(self.eval_steps, self.eval_losses, 
                    label='Validation Loss', color='red', linewidth=2, marker='o', markersize=6)
        
        plt.title('Andamento della Loss durante il Training', fontsize=16, fontweight='bold')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Aggiungi annotazione con valori finali
        final_train_loss = self.train_losses[-1] if self.train_losses else None
        final_eval_loss = self.eval_losses[-1] if self.eval_losses else None
        
        info_text = f"Training: {len(self.train_losses)} steps"
        if final_train_loss:
            info_text += f"\nFinal Train Loss: {final_train_loss:.4f}"
        if final_eval_loss:
            info_text += f"\nFinal Val Loss: {final_eval_loss:.4f}"
        
        plt.annotate(info_text, 
                    xy=(0.02, 0.98), 
                    xycoords='axes fraction',
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Salva il grafico
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f'loss_plot_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Grafico della loss salvato in: {save_path}")
        
        plt.show()
        plt.close()


class QualityRegressionDataset(Dataset):
    """
    Dataset PyTorch per il task di regressione della qualità.
    Carica dati in formato JSONL (un JSON per riga) e li converte in tensori
    compatibili con il Trainer di HuggingFace.
    
    Ogni sample JSONL deve contenere almeno:
    - 'content': il testo del post
    - 'quality_score' (float 0-1) OPPURE 'quality_level' ('high'/'medium'/'low')
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        
        # Carica il file JSONL riga per riga (memory-efficient per dataset grandi)
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        self.tokenizer = tokenizer
        # max_length=512 è il limite standard di BERT: sequenze più lunghe vengono troncate
        self.max_length = max_length
        print(f" Caricati {len(self.samples)} sample da {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get('content', '')
        
        # Supporta due formati di label:
        # 1. quality_score: valore numerico continuo [0, 1] (preferito per regressione)
        # 2. quality_level: label categorica mappata a valori discreti
        if 'quality_score' in sample:
            score = float(sample['quality_score'])
        else:
            level_to_score = {"high": 0.85, "medium": 0.50, "low": 0.20}
            score = level_to_score.get(sample.get('quality_level', 'medium'), 0.50)
        
        # Tokenizzazione: converte il testo in input_ids (indici del vocabolario BERT)
        # e attention_mask (1 per token reali, 0 per padding)
        # truncation=True taglia i testi più lunghi di max_length
        # padding='max_length' aggiunge padding fino a max_length per avere batch uniformi
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float)
        }


class QualityRegressionModel(nn.Module):
    """
    Modello di regressione basato su BERT italiano (dbmdz/bert-base-italian-uncased).
    
    Architettura:
        BERT Encoder (12 layer transformer) -> CLS token embedding (768-dim)
        -> Dropout -> Linear(768, 1) -> Sigmoid -> score ∈ [0, 1]
    
    Il modello sfrutta il transfer learning: BERT è pre-addestrato su testo italiano
    e viene adattato (fine-tuned) al task specifico di quality scoring.
    La sigmoid finale vincola l'output nell'intervallo [0, 1], interpretabile
    come probabilità/punteggio di qualità.
    """
    
    def __init__(self, model_name: str = "dbmdz/bert-base-italian-uncased", dropout: float = 0.1):
        super().__init__()
        # BERT base ha 12 layer transformer, hidden_size=768, ~110M parametri
        self.bert = AutoModel.from_pretrained(model_name)
        # Dropout per regolarizzazione: spegne casualmente il 10% dei neuroni
        # durante il training per prevenire overfitting
        self.dropout = nn.Dropout(dropout)
        # Testa di regressione: proietta il vettore CLS (768-dim) in un singolo scalare
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        # Sigmoid comprime l'output in [0, 1] per interpretarlo come punteggio di qualità
        self.sigmoid = nn.Sigmoid()
        
        # Strategia di Partial Freezing (congelamento parziale):
        # Congela TUTTI i parametri di BERT (embedding + 12 layer) per preservare
        # le rappresentazioni linguistiche apprese durante il pre-training
        for param in self.bert.parameters():
            param.requires_grad = False
        # Poi scongela SOLO gli ultimi 4 layer del transformer encoder.
        # I layer più alti codificano features più astratte e task-specific,
        # quindi vanno adattati al nostro task di quality scoring.
        for layer in self.bert.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass compatibile con il Trainer di HuggingFace.
        
        Il Trainer si aspetta che forward() restituisca:
        - (loss, logits) durante il training (quando labels è fornito)
        - (None, logits) durante l'inferenza
        
        **kwargs cattura eventuali argomenti extra passati dal Trainer
        (es. token_type_ids) che non ci servono, evitando errori.
        """
        # BERT restituisce un oggetto con last_hidden_state di shape (batch, seq_len, 768) 
        # (dove batch è il numero di campioni per batch, seq_len è il numero di token di ogni campione, 768 è la dimensione dei vettori di embedding dei token)
        # Prendiamo solo il primo token di ogni sequenza (il token [CLS]) dall'output dell'ultimo layer dell'encoder.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Dropout applicato solo durante il training (disattivato automaticamente in eval)
        cls_output = self.dropout(cls_output)
        # Proiezione lineare: (batch, 768) -> (batch, 1)
        score = self.regressor(cls_output)
        score = self.sigmoid(score).squeeze() # squeeze() rimuove la dimensione 1, quindi (batch, 1) -> (batch,), cioè uno scalare per ogni elemento del batch
        
        # Per il Trainer: restituisce una tupla (loss, logits) o un dizionario
        if labels is not None:
            # unsqueeze(0) riaggiunge la dimensione batch perché MSELoss() richiede tensori 1-dim.
            if score.dim() == 0:
                score = score.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            loss = nn.MSELoss()(score, labels)
            return (loss, score)
        
        return (None, score)


class ScoreAgent:
    """
    Agente per la valutazione della qualità dei post
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        # self.device decide quale elaboratore useremo, in base a cosa ha il computer (o CPU o GPU se disponibile)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu') 
        # Carichiamo il tokenizer per l'italiano (dbmdz/bert-base-italian-uncased)
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
        # self.loss_history tiene traccia della loss (errore) durante l'addestramento
        self.loss_history = {'train': [], 'val': []}
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self.model = None
            print(" Nessun modello fine-tunato trovato.")
        
        print(f" ScoreAgent inizializzato su {self.device}")
    
    def _load_model(self, model_path: str):
        """Carica modello fine-tunato"""
        self.model = QualityRegressionModel()
        # Carica i pesi (supporta sia pytorch_model.bin che model.safetensors)
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        safetensors_path = os.path.join(model_path, "model.safetensors")
        
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device=self.device)
            print(" Trovato modello in formato safetensors, caricamento in corso...")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location=self.device)
            print(" Trovato modello in formato PyTorch bin, caricamento in corso...")
        else:
            raise FileNotFoundError(f"Nessun file di pesi del modello trovato in {model_path}")
            
        # Carica i pesi nel modello
        self.model.load_state_dict(state_dict)
        # Sposta il modello sulla CPU o GPU
        self.model.to(self.device)
        # Mette il modello in modalità valutazione (disattiva dropout, ecc.)
        self.model.eval()
        print(f" Modello caricato da {model_path}")
    
    def train(self, train_path: str, val_path: str = None, output_dir: str = "models/score_agent",
              epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5,
              plot_loss: bool = True):
        """
        Addestra il modello di regressione della qualità
        
        Args:
            train_path: Percorso al file di training (jsonl)
            val_path: Percorso al file di validazione (jsonl)
            output_dir: Directory dove salvare il modello
            epochs: Numero di epoche
            batch_size: Batch size
            learning_rate: Learning rate
            plot_loss: Se True, mostra il grafico della loss
        """
        print("\n" + "="*60)
        print(" Fine-tuning BERT italiano per Quality Scoring")
        print("="*60)
        
        # Prepara dataset
        train_dataset = QualityRegressionDataset(train_path, self.tokenizer)
        
        val_dataset = None
        if val_path and os.path.exists(val_path):
            val_dataset = QualityRegressionDataset(val_path, self.tokenizer)
        
        # Inizializza il modello con i pesi pre-addestrati di BERT italiano
        # e la testa di regressione con pesi casuali (da addestrare)
        self.model = QualityRegressionModel()
        self.model.to(self.device)
        
        # Callback per registrare la loss ad ogni step di training
        loss_callback = LossLoggingCallback()
        
        # Metriche di valutazione calcolate a fine epoca sul validation set
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred # eval_pred è una tupla contenente le predizioni del modello e le etichette vere
            # Clip a [0, 1]: la sigmoid dovrebbe già garantirlo, ma evitiamo
            # artefatti numerici nei calcoli delle metriche
            predictions = np.clip(predictions, 0, 1)
            mse = mean_squared_error(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            acc_within_01 = np.mean(np.abs(predictions - labels) < 0.1) #percentuale di predizioni con errore entro 0.1
            
            return {
                'mse': mse,
                'mae': mae,
                'acc_within_0.1': acc_within_01
            }
        
        # Configurazione degli iperparametri di training con il Trainer di HuggingFace
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            # Warmup: per i primi 100 step il learning rate sale gradualmente da 0
            # al valore target. Questo stabilizza il training iniziale evitando
            # aggiornamenti troppo aggressivi prima che il modello si "orienti"
            warmup_steps=100,
            # Weight decay (L2 regularization): penalizza pesi grandi per
            # prevenire overfitting. 0.01 è il valore standard per BERT
            weight_decay=0.01,
            logging_dir='./logs',
            # Logga metriche ogni 10 step per monitorare l'andamento del training
            logging_steps=10,
            # Valuta sul validation set a fine epoca per monitorare overfitting
            eval_strategy="epoch" if val_dataset else "no",
            # Salva un checkpoint a fine epoca per poter riprendere il training
            save_strategy="epoch" if val_dataset else "no",
            # Carica automaticamente il modello con la loss più bassa sul validation set
            # a fine training (early stopping implicito) recuperando da checkpoint l' epoch che ha dato i risultati migliori
            load_best_model_at_end=True if val_dataset else False,
            # Usa MSE come metrica per selezionare il miglior modello
            metric_for_best_model="mse",
            # Per MSE, valori più bassi sono migliori
            greater_is_better=False,
            learning_rate=learning_rate,
            # report_to=[] disabilita logging su piattaforme esterne (WandB, TensorBoard)
            report_to=[],
        )
        
        # Viene passato al Trainer anche il compute_metrics in modo che lo richiami automaticamente alla fine di ogni epoca 
        # per calcolare le metriche sul validation set, e anche la loss_callback per salvare la loss ad ogni step
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None,
            callbacks=[loss_callback] if plot_loss else [],
        )
        
        # Training
        print("\n Inizio training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"   Validation samples: {len(val_dataset)}")
        print("-" * 40)
        
        trainer.train()
        
        # Salva modello
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"\n Modello salvato in {output_dir}")
        
        # Report finale
        if val_dataset:
            eval_results = trainer.evaluate()
            print("\n Risultati valutazione:")
            print(f"   MSE: {eval_results.get('eval_mse', 'N/A'):.4f}")
            print(f"   MAE: {eval_results.get('eval_mae', 'N/A'):.4f}")
            print(f"   Accuracy within 0.1: {eval_results.get('eval_acc_within_0.1', 'N/A'):.2%}")
        
        # Plot della loss
        if plot_loss:
            print("\n Generazione grafico della loss...")
            loss_callback.plot_losses(output_dir)
        
        print("="*60)
        return trainer
    
    def _fallback_score_only(self, post_content: str) -> Dict:
        """
        Calcola lo score di qualità tramite euristica rule-based (senza BERT).
        
        Questo metodo viene usato come fallback quando il modello BERT non è
        disponibile (non ancora addestrato).
        """

        # ── 1. Lunghezza (max 0.35) ──
        word_count = len(post_content.split())
        if word_count < 300:
            length_score = 0.0
        elif word_count < 700:
            length_score = 0.10
        elif word_count < 1500:
            length_score = 0.20
        elif word_count < 3000:
            length_score = 0.30
        elif word_count <= 5500:
            length_score = 0.35  # range ideale per il blog
        else:
            length_score = 0.25  # troppo lungo, lieve penalità

        # ── 2. Struttura (max 0.35) ──
        # Conta heading (# , ## , ### )
        headings = len(re.findall(r'^#{1,3}\s+.+', post_content, re.MULTILINE))
        heading_score = min(headings * 0.05, 0.15)  # max 0.15 con 3+ heading

        # Conta paragrafi non vuoti (blocchi di testo separati da righe vuote)
        paragraphs = [p.strip() for p in post_content.split('\n\n') if p.strip()]
        para_score = min(len(paragraphs) * 0.02, 0.10)  # max 0.10 con 5+ paragrafi

        # Presenza di liste puntate o numerate
        has_lists = bool(re.search(r'^[\s]*[-*•]\s+|^\s*\d+\.\s+', post_content, re.MULTILINE))
        list_score = 0.10 if has_lists else 0.0

        structure_score = heading_score + para_score + list_score

        # ── 3. Citazioni (max 0.15) ──
        # Conta il numero totale di citazioni trovate nel testo
        citation_patterns = [
            r'\[Source:[^\]]*\]',     # formato richiesto: [Source: Title]
            r'\[Fonte:[^\]]*\]',      # variante italiana: [Fonte: Titolo]
            r'https?://\S+',          # URL inline
        ]
        citation_count = sum(len(re.findall(p, post_content, re.IGNORECASE)) for p in citation_patterns)
        citation_score = min(citation_count * 0.03, 0.15)  # max 0.15 con 5+ citazioni

        # ── 4. Base minima (0.15) ──
        # Ogni post generato merita almeno un punto di partenza
        base_score = 0.15

        # ── Totale ──
        score = base_score + length_score + structure_score + citation_score
        score = min(max(score, 0.0), 1.0)

        return {
            'quality_score': round(score, 3),
            'quality_label': self._score_to_label(score),
            'method': 'fallback_heuristic',
            'details': {
                'word_count': word_count,
                'base_score': base_score,
                'length_score': round(length_score, 3),
                'structure_score': round(structure_score, 3),
                'citation_score': round(citation_score, 3),
                'headings_found': headings,
                'paragraphs_found': len(paragraphs),
                'has_lists': has_lists,
                'citation_count': citation_count
            }
        }
    
    @traceable(name="ScoreAgent-ScoreWithComparison", run_type="chain", tags=["score", "comparison"])
    def score_with_comparison(self, post_content: str) -> Dict:
        """
        Valuta la qualità calcolando ENTRAMBI gli score (fallback euristica E BERT)
        per permettere un confronto diagnostico tra i due metodi.
        Lo score finale usato per le decisioni del workflow è quello BERT
        (se disponibile), altrimenti il fallback euristico.
        """
        # 1. Calcola sempre lo score euristico (baseline interpretabile)
        fallback_result = self._fallback_score_only(post_content)
        
        # 2. Calcola BERT score solo se il modello fine-tunato è disponibile
        if self.model is None:
            bert_result = None
        else:
            # Tokenizza il testo di input per BERT
            inputs = self.tokenizer(
                post_content,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            # Sposta i tensori ottenuti dal tokenizer sul device corretto (CPU/GPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # torch.no_grad() disabilita il calcolo dei gradienti durante l'inferenza:
            # risparmia memoria e velocizza l'esecuzione (non serve backpropagation)
            with torch.no_grad():
                _, score = self.model(**inputs) # richiama automaticamente il forward del modello
                if isinstance(score, torch.Tensor):
                    score = score.item() # .item() converte un tensore scalare in un float Python
            
            # Clamp di sicurezza: la sigmoid dovrebbe già restituire [0,1]
            # ma evitiamo artefatti numerici
            score = min(max(score, 0.0), 1.0)
            
            bert_result = {
                'quality_score': round(score, 3), # arrotonda lo score a 3 cifre decimali
                'quality_label': self._score_to_label(score), # converte lo score in label (low, medium, high)
                'method': 'bert_finetuned' # metodo utilizzato per calcolare lo score
            }
        
        # 3. Stampa comparativa per debug e analisi
        self._print_comparison(fallback_result, bert_result)
        
        # 4. Priorità a BERT: ha appreso pattern semantici dal dataset di training,
        # mentre il fallback valuta solo features superficiali/strutturali
        return bert_result if bert_result else fallback_result
    
    def _print_comparison(self, fallback: Dict, bert: Dict = None):
        """
        Stampa comparazione tra score fallback e BERT
        """
        print("\n" + "="*60)
        print(" QUALITY SCORE COMPARISON")
        print("="*60)
        
        print(f"\n FALLBACK (euristica):")
        print(f"   Score: {fallback['quality_score']}")
        print(f"   Label: {fallback['quality_label']}")
        if 'details' in fallback:
            d = fallback['details']
            print(f"   Word count: {d.get('word_count', 'N/A')}")
            print(f"   Base score:      {d.get('base_score', 'N/A')}")
            print(f"   Length score:     {d.get('length_score', 'N/A')}")
            print(f"   Structure score:  {d.get('structure_score', 'N/A')}  "
                  f"(headings={d.get('headings_found', 0)}, "
                  f"paragraphs={d.get('paragraphs_found', 0)}, "
                  f"lists={'yes' if d.get('has_lists') else 'no'})")
            print(f"   Citation score:   {d.get('citation_score', 'N/A')}  "
                  f"(citations found={d.get('citation_count', 0)})")
        
        if bert:
            print(f"\n BERT (fine-tuned):")
            print(f"   Score: {bert['quality_score']}")
            print(f"   Label: {bert['quality_label']}")
            
            # Differenza tra i due metodi
            diff = abs(fallback['quality_score'] - bert['quality_score'])
            diff_icon = "🟢" if diff < 0.2 else "🟡" if diff < 0.4 else "🔴"
            print(f"\n{diff_icon} Difference: {diff:.3f} ({diff*100:.1f}%)")
        
        print("="*60)
    
    @traceable(name="ScoreAgent-Score", run_type="chain", tags=["score"])
    def score(self, post_content: str, print_comparison: bool = True) -> Dict:
        """
        Valuta la qualità di un singolo post
        
        Args:
            post_content: Testo del post da valutare
            print_comparison: Se True, stampa anche il fallback per confronto
        """
        if print_comparison:
            return self.score_with_comparison(post_content)
        
        # Versione semplice (fallback)
        if self.model is None:
            return self._fallback_score_only(post_content)
        
        inputs = self.tokenizer(
            post_content,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _, score = self.model(**inputs)
            if isinstance(score, torch.Tensor):
                score = score.item()
        
        score = min(max(score, 0.0), 1.0)
        
        return {
            'quality_score': round(score, 3),
            'quality_label': self._score_to_label(score),
            'method': 'bert_finetuned'
        }
    
    def _score_to_label(self, score: float) -> str:
        """Converte lo score numerico in una label categorica leggibile."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "acceptable"
        elif score >= 0.2:
            return "poor"
        else:
            return "unacceptable"


@traceable(name="ScoreAgent", run_type="chain", tags=["agent", "score"])
def score_agent(state):
    """
    Nodo del workflow LangGraph per la valutazione della qualità del post.
    
    Flusso:
    1. Cerca il modello BERT fine-tunato su disco (formati .bin o .safetensors)
    2. Se trovato, usa BERT per scoring semantico; altrimenti usa il fallback euristico
    3. Applica soglie per decidere se il post passa il quality check
    """
    print("\n" + "="*50)
    print(" SCORE AGENT - Quality Assessment")
    print("="*50)
    
    # Cerca il modello fine-tunato nella directory predefinita.
    # Il Trainer di HuggingFace salva i pesi in formato safetensors (default recente)
    # o pytorch_model.bin (formato legacy)
    model_path = "models/score_agent"
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    
    if os.path.exists(model_path) and (os.path.exists(bin_path) or os.path.exists(safetensors_path)):
        scorer = ScoreAgent(model_path=model_path)
    else:
        # Senza modello BERT, lo ScoreAgent userà automaticamente il fallback
        # euristico (rule-based) per valutare la qualità
        print(" Modello non trovato, uso fallback euristico")
        scorer = ScoreAgent()
    
    draft = state.get('draft_post', {})
    content = draft.get('content', '')
    
    if not content:
        print(" Nessun contenuto da valutare")
        return {'quality_evaluation': {'quality_score': 0.5}, 'quality_passed': True}
    
    # Calcola e stampa ENTRAMBI gli score (fallback + BERT) per diagnostica,
    # poi usa BERT come score decisionale (se disponibile)
    evaluation = scorer.score(content, print_comparison=True)
    
    print(f"\n Final decision (using: {evaluation['method']})")
    print(f"   Score: {evaluation['quality_score']}")
    print(f"   Label: {evaluation['quality_label']}")
    threshold = 0.70
    quality_passed = evaluation['quality_score'] >= threshold
    barely_passed = evaluation['quality_score'] >= 0.60 and evaluation['quality_score'] < threshold

    print("="*50)

    return {
        'quality_passed': quality_passed,
        'barely_passed': barely_passed
    }


if __name__ == "__main__":
    import argparse
    
    # Usiamo ArgumentParser per definire i parametri da linea di comando
    # Questi permettono di configurare l'addestramento senza modificare il codice
    parser = argparse.ArgumentParser(description='Train Score Agent')
    parser.add_argument('--train', type=str, default='data/bert_dataset/train.jsonl',
                        help='Path to training data')
    parser.add_argument('--val', type=str, default='data/bert_dataset/val.jsonl',
                        help='Path to validation data')
    parser.add_argument('--output', type=str, default='models/score_agent',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable loss plotting')
    
    args = parser.parse_args()
    
    scorer = ScoreAgent()
    scorer.train(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        plot_loss=not args.no_plot
    )