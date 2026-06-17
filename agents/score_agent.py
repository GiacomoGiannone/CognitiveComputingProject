# agents/score_agent.py
"""
Score Agent basato su BERT italiano (dbmdz/bert-base-italian-uncased)
Valuta la qualità di un post con un punteggio da 0 a 1.
Fine-tuning per regressione su dataset di post etichettati.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from typing import List, Dict
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class QualityRegressionDataset(Dataset):
    """Dataset per regressione della qualità"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"📊 Caricati {len(self.samples)} sample da {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get('content', '')
        
        if 'quality_score' in sample:
            score = float(sample['quality_score'])
        else:
            level_to_score = {"high": 0.85, "medium": 0.50, "low": 0.20}
            score = level_to_score.get(sample.get('quality_level', 'medium'), 0.50)
        
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
    BERT italiano con testa di regressione per predire qualità 0-1
    """
    
    def __init__(self, model_name: str = "dbmdz/bert-base-italian-uncased", dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Freezing: solo gli ultimi 4 layer
        for param in self.bert.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass compatibile con Trainer di transformers
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        score = self.regressor(cls_output)
        score = self.sigmoid(score).squeeze()
        
        # Per il Trainer: restituisce una tupla (loss, logits) o un dizionario
        if labels is not None:
            # Assicura che le dimensioni corrispondano
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
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self.model = None
            print("⚠️ Nessun modello fine-tunato trovato.")
        
        print(f"📊 ScoreAgent inizializzato su {self.device}")
    
    def _load_model(self, model_path: str):
        """Carica modello fine-tunato"""
        self.model = QualityRegressionModel()
        # Carica i pesi (supporta sia pytorch_model.bin che model.safetensors)
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        safetensors_path = os.path.join(model_path, "model.safetensors")
        
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device=self.device)
            print("📂 Trovato modello in formato safetensors, caricamento in corso...")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location=self.device)
            print("📂 Trovato modello in formato PyTorch bin, caricamento in corso...")
        else:
            raise FileNotFoundError(f"Nessun file di pesi del modello trovato in {model_path}")
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Modello caricato da {model_path}")
    
    def train(self, train_path: str, val_path: str = None, output_dir: str = "models/score_agent",
              epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
        """
        Addestra il modello di regressione della qualità
        """
        print("\n" + "="*60)
        print("🏋️ Fine-tuning BERT italiano per Quality Scoring")
        print("="*60)
        
        # Prepara dataset
        train_dataset = QualityRegressionDataset(train_path, self.tokenizer)
        
        val_dataset = None
        if val_path and os.path.exists(val_path):
            val_dataset = QualityRegressionDataset(val_path, self.tokenizer)
        
        # Inizializza modello
        self.model = QualityRegressionModel()
        self.model.to(self.device)
        
        # Metriche di valutazione
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.clip(predictions, 0, 1)
            mse = mean_squared_error(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            acc_within_01 = np.mean(np.abs(predictions - labels) < 0.1)
            
            return {
                'mse': mse,
                'mae': mae,
                'acc_within_0.1': acc_within_01
            }
        
        # Configurazione training
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="mse",
            greater_is_better=False,
            learning_rate=learning_rate,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None,
        )
        
        # Training
        print("\n🚀 Inizio training...")
        trainer.train()
        
        # Salva modello
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"\n✅ Modello salvato in {output_dir}")
        
        # Report finale
        if val_dataset:
            eval_results = trainer.evaluate()
            print("\n📊 Risultati valutazione:")
            print(f"   MSE: {eval_results.get('eval_mse', 'N/A'):.4f}")
            print(f"   MAE: {eval_results.get('eval_mae', 'N/A'):.4f}")
            print(f"   Accuracy within 0.1: {eval_results.get('eval_acc_within_0.1', 'N/A'):.2%}")
        
        print("="*60)
        return trainer
    
    def _fallback_score_only(self, post_content: str) -> Dict:
        """
        Calcola SOLO lo score di fallback (euristica) senza BERT
        """
        word_count = len(post_content.split())
        
        score = 0.5
        if 500 <= word_count <= 1500:
            score += 0.1
        elif word_count < 200:
            score -= 0.2
        
        if '[Source:' in post_content or 'fonte' in post_content.lower():
            score += 0.1
        
        score = min(max(score, 0.0), 1.0)
        
        return {
            'quality_score': round(score, 3),
            'quality_label': self._score_to_label(score),
            'confidence': 0.5,
            'method': 'fallback_heuristic',
            'details': {
                'word_count': word_count,
                'has_citations': '[Source:' in post_content or 'fonte' in post_content.lower(),
                'length_bonus': 0.1 if (500 <= word_count <= 1500) else (-0.2 if word_count < 200 else 0)
            }
        }
    
    def score_with_comparison(self, post_content: str) -> Dict:
        """
        Valuta la qualità RESTITUENDO ENTRAMBI gli score (fallback E BERT)
        """
        # Calcola fallback score
        fallback_result = self._fallback_score_only(post_content)
        
        # Calcola BERT score
        if self.model is None:
            bert_result = None
        else:
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
            
            bert_result = {
                'quality_score': round(score, 3),
                'quality_label': self._score_to_label(score),
                'confidence': self._estimate_confidence(score),
                'method': 'bert_finetuned'
            }
        
        # Stampa comparativa
        self._print_comparison(fallback_result, bert_result)
        
        # Sceglie BERT se disponibile, altrimenti fallback
        return bert_result if bert_result else fallback_result
    
    def _print_comparison(self, fallback: Dict, bert: Dict = None):
        """
        Stampa comparazione tra score fallback e BERT
        """
        print("\n" + "="*60)
        print("📊 QUALITY SCORE COMPARISON")
        print("="*60)
        
        print(f"\n🔧 FALLBACK (euristica):")
        print(f"   Score: {fallback['quality_score']}")
        print(f"   Label: {fallback['quality_label']}")
        print(f"   Confidence: {fallback['confidence']}")
        if 'details' in fallback:
            print(f"   Details: word_count={fallback['details']['word_count']}, "
                  f"has_citations={fallback['details']['has_citations']}")
        
        if bert:
            print(f"\n🤖 BERT (fine-tuned):")
            print(f"   Score: {bert['quality_score']}")
            print(f"   Label: {bert['quality_label']}")
            print(f"   Confidence: {bert['confidence']}")
            
            # Differenza tra i due metodi
            diff = abs(fallback['quality_score'] - bert['quality_score'])
            diff_icon = "🟢" if diff < 0.1 else "🟡" if diff < 0.2 else "🔴"
            print(f"\n{diff_icon} Difference: {diff:.3f} ({diff*100:.1f}%)")
            
            # Quale metodo ha più confidence?
            if fallback['confidence'] > bert['confidence']:
                print(f"   📌 Fallback ha confidence maggiore ({fallback['confidence']:.0%} vs {bert['confidence']:.0%})")
            else:
                print(f"   📌 BERT ha confidence maggiore ({bert['confidence']:.0%} vs {fallback['confidence']:.0%})")
        else:
            print(f"\n⚠️ BERT model not available, using fallback only")
        
        print("="*60)
    
    def score(self, post_content: str, print_comparison: bool = True) -> Dict:
        """
        Valuta la qualità di un singolo post
        
        Args:
            post_content: Testo del post da valutare
            print_comparison: Se True, stampa anche il fallback per confronto
        """
        if print_comparison:
            return self.score_with_comparison(post_content)
        
        # Versione semplice (solo BERT o fallback)
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
            'confidence': self._estimate_confidence(score),
            'method': 'bert_finetuned'
        }
    
    def _score_to_label(self, score: float) -> str:
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
    
    def _estimate_confidence(self, score: float) -> float:
        confidence = 0.5 + abs(score - 0.5)
        return round(min(confidence, 0.95), 3)


def score_agent(state):
    """Integrazione nel workflow LangGraph"""
    print("\n" + "="*50)
    print("🎯 SCORE AGENT - Quality Assessment")
    print("="*50)
    
    model_path = "models/score_agent"
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    
    if os.path.exists(model_path) and (os.path.exists(bin_path) or os.path.exists(safetensors_path)):
        scorer = ScoreAgent(model_path=model_path)
    else:
        print("⚠️ Modello non trovato, uso fallback euristico")
        scorer = ScoreAgent()
    
    draft = state.get('draft_post', {})
    content = draft.get('content', '')
    
    if not content:
        print("⚠️ Nessun contenuto da valutare")
        return {'quality_evaluation': {'quality_score': 0.5}, 'quality_passed': True}
    
    # Ora stampa ENTRAMBI gli score (fallback + BERT)
    evaluation = scorer.score(content, print_comparison=True)
    
    print(f"\n✅ Final decision (using: {evaluation['method']})")
    print(f"   Score: {evaluation['quality_score']}")
    print(f"   Label: {evaluation['quality_label']}")
    
    quality_passed = evaluation['quality_score'] >= 0.4
    
    if not quality_passed:
        print(f"\n⚠️ Qualità insufficiente (score < 0.4), richiesta rigenerazione...")
    
    print("="*50)
    
    return {
        'quality_evaluation': evaluation,
        'quality_passed': quality_passed,
        'requires_regeneration': not quality_passed
    }


if __name__ == "__main__":
    import argparse
    
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
    
    args = parser.parse_args()
    
    scorer = ScoreAgent()
    scorer.train(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )