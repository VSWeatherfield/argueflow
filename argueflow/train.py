from torch.nn import CrossEntropyLoss


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = CrossEntropyLoss()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
