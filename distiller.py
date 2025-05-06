import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load smaller teacher and student models
teacher_model = AutoModelForCausalLM.from_pretrained("gpt2")  # NOT gpt2-large (too big for CPU)
student_model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Fix pad token
tokenizer.pad_token = tokenizer.eos_token

# 3. Move models to CPU
device = torch.device("cpu")
teacher_model.to(device)
student_model.to(device)

# 4. Define dummy dataset
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Machine learning models can be trained to predict outcomes.",
    "Natural language processing enables machines to understand text.",
    "Transformers are powerful architectures for sequence modeling."
]

# 5. Distillation loss function
def distillation_loss(student_logits, teacher_logits, target_ids, alpha=0.5, temperature=2.0):
    loss_ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        target_ids.view(-1)
    )
    loss_kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)
    return alpha * loss_ce + (1 - alpha) * loss_kd

# 6. Simulate training loop + record loss
losses = []
for epoch in range(5):
    total_loss = 0
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits

        student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = distillation_loss(student_logits, teacher_logits, input_ids)
        total_loss += loss.item()

    avg_loss = total_loss / len(texts)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1} - Distillation Loss: {avg_loss:.4f}")

# 7. Plot loss curve
plt.plot(range(1, 6), losses, marker='o')
plt.title("Distillation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Final inference comparison
prompt = "The future of artificial intelligence is"
gen_inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    teacher_gen = teacher_model.generate(**gen_inputs, max_length=50)
    student_gen = student_model.generate(**gen_inputs, max_length=50)

teacher_text = tokenizer.decode(teacher_gen[0], skip_special_tokens=True)
student_text = tokenizer.decode(student_gen[0], skip_special_tokens=True)

print("\n🧠 Teacher Model Output:\n", teacher_text)
print("\n📘 Student Model Output:\n", student_text)
