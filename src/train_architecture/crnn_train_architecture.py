import torch
from tqdm import tqdm

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# encodes names-strings into integer indices
def encode_targets(labels):  # labels = ["AB12", "XYZ"]
    flat, lengths = [], []
    for s in labels:
        flat.extend(CHARS.index(c) + 1 for c in s)  # +1 because blank=0
        lengths.append(len(s))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()  # enables model to train (just changes flag)
    total = 0.0  # accumulates loss for every batch in epoch. interface/debug

    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)  # puts images into GPU memmory, so model(images) are performed using GPU
        log_probs = model(images)  # forward pass
        T, B, _ = log_probs.shape  # (T, B, C)

        targets, target_lengths = encode_targets(labels)  # makes one_hot from plain text
        input_lengths = torch.full((B,), T, dtype=torch.long)  # tells torch timesteps of sample

        loss = criterion(log_probs.cpu(), targets, input_lengths, target_lengths)

        optimizer.zero_grad()  # cleans gradient for correct gradient descent
        loss.backward()  # pytorch computes gradient using backpropagation
        optimizer.step()  # makes one step of gradient descent
        total += loss.item()  # converts tensor into float value

    return total / len(loader)  # convention


# decods one image. from CTC results to actual label
def decode_image(log_probs):  # (32, B, 37)
    idx = log_probs.argmax(2)  # (32, B) - value is index
    out = []
    for b in range(idx.shape[1]):  # loop for batches
        seq, prev = [], -1  # -1 so prev != first
        for t in range(idx.shape[0]):  # loop for timesteps
            i = idx[t, b].item()  # extracts value
            if i != prev and i != 0:
                seq.append(CHARS[i - 1])
            prev = i
        out.append("".join(seq))
    return out


# decods only one batch, so it is much faster (for debug)
# prints only first name
def decode_batch(model, loader):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(loader))  # iter - iterator for loader, next - first batch
        preds = decode_image(model(images))
        print(f"prediction: {preds[0]}, actual: {labels[0]}")

    return preds


def validation(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:  # loop across batches
            images = images.to(device)
            log_probs = model(images)
            T, B, _ = log_probs.shape

            targets, target_lengths = encode_targets(labels)
            input_lengths = torch.full((B,), T, dtype=torch.long)
            loss = criterion(log_probs.cpu(), targets, input_lengths,
                             target_lengths)  # log_probs.cpu() copies data from GPU to RAM

            total += loss.item()  # already normalized loss across images in batch (mean)

    return total / len(loader)  # sum of loss of all batches / number of batches

# for more visually understandable information during training
def test_model(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_symbols, correct_symbols = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)
            pred = decode_image(log_probs.cpu())

            for p, s in zip(pred, labels):
                total += 1
                if p == s:
                    correct += 1

                correct_symbols += sum(a == b for a, b in zip(p, s))
                total_symbols += len(s)

    return correct / total, correct_symbols / total_symbols

# for final testing on test_dataset
def test_model_debug(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_symbols, correct_symbols = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            log_probs = model(images)
            pred = decode_image(log_probs)

            for p, s in zip(pred, labels):
                total += 1
                if p == s:
                    correct += 1
                else:
                    print(f"prediction: {p}, actual: {s}")

                correct_symbols += sum(a == b for a, b in zip(p, s))
                total_symbols += len(s)

    return correct / total, correct_symbols / total_symbols