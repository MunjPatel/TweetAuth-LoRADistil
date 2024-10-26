import torch
import pandas as pd
import time
import datetime
import numpy as np
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from lora_layers import apply_lora_to_model

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("train.csv")
tweets = df.text.values
labels = df.target.values

# Load pre-trained DistilBERT model with classification head
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
for param in model.parameters():
    param.requires_grad = False  # Freeze model parameters


apply_lora_to_model(model)

# Tokenize tweets
def tokenize_tweets(tokenizer, tweets, max_length):
    input_ids = []
    attention_masks = []

    for tweet in tweets:
        encoded_dict = tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',  # Faster than pad_to_max_length
            truncation=True,  # For ensuring the length constraint
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Determine the max sequence length
max_seq_len = max([len(tokenizer.encode(sent, add_special_tokens=True)) for sent in tweets])

# Tokenize all tweets
input_ids, attention_masks = tokenize_tweets(tokenizer, tweets, max_length=max_seq_len)

# Prepare dataset
labels_tensor = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_masks, labels_tensor)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
def create_dataloaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    return train_loader, val_loader

train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset)

# Move model to device (GPU/CPU)
model = model.to(device)

# Optimizer and scheduler setup
def setup_optimizer_scheduler(model, learning_rate=2e-5, epsilon=1e-8, epochs=2):
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return optimizer, scheduler

optimizer, scheduler = setup_optimizer_scheduler(model)

# Helper function: accuracy calculation
def calculate_accuracy(predictions, labels):
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Time formatting utility
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

# Training and validation process
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=2):
    total_t0 = time.time()

    training_stats = []
    best_eval_accuracy = 0

    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch + 1} / {epochs} ========")
        print("Training...")

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            optimizer.zero_grad()

            output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")

        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            with torch.no_grad():
                output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = output.loss
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_loss += loss.item()
            total_eval_accuracy += calculate_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)

        validation_time = format_time(time.time() - t0)
        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")

        # Save best model
        if avg_val_accuracy > best_eval_accuracy:
            torch.save(model, 'best_lora_distilbert_model.pth')
            best_eval_accuracy = avg_val_accuracy

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss,
            'Validation Accuracy': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")

train_model(model, train_dataloader, val_dataloader, optimizer, scheduler)
