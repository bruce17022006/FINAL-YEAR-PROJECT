
from pathlib import Path
import torch

from train_stage1_asr import StudentAcousticModel

CHECKPOINTS = [
    Path("checkpoints_stage1/best_student_asr.pt"),
    Path("checkpoints_stage1/best_student_asr_corrected.pt"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = StudentAcousticModel(
    input_dim=80,
    encoder_dim=256,
    num_encoder_layers=4,
    vocab_size=1024,
    teacher_dim=2048,
    blank_id=4,
)

expected = model.state_dict()

print("Device:", device)
print("Model parameter tensors:", len(expected))

for path in CHECKPOINTS:
    print("\n" + "=" * 50)
    print("Checkpoint:", path)

    if not path.is_file():
        print("NOT FOUND")
        continue

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(state, dict):
        print("Unexpected checkpoint format:", type(state))
        continue

    # Handle checkpoints that wrap the weights in a dictionary.
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model_state_dict" in state and isinstance(
        state["model_state_dict"], dict
    ):
        state = state["model_state_dict"]

    matched = []
    missing = []
    shape_mismatch = []

    for key, tensor in expected.items():
        if key not in state:
            missing.append(key)
        elif not hasattr(state[key], "shape") or (
            tuple(state[key].shape) != tuple(tensor.shape)
        ):
            shape_mismatch.append(key)
        else:
            matched.append(key)

    unexpected = [key for key in state if key not in expected]

    print("Matched tensors:", len(matched), "/", len(expected))
    print("Missing tensors:", len(missing))
    print("Shape mismatches:", len(shape_mismatch))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("Missing examples:", missing[:5])
    if shape_mismatch:
        print("Shape mismatch examples:", shape_mismatch[:5])
    if unexpected:
        print("Unexpected examples:", unexpected[:5])

    if len(matched) == len(expected) and not unexpected:
        print("RESULT: Architecture and weight shapes match.")
    else:
        print("RESULT: Investigate the reported differences.")
