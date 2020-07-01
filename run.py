"""
모델 성능 재현을 위한 스크립트입니다.

하지만 재현 가능성을 테스트해보지 않은 관계로, README.md에 있는 프로세스를 따라가는 것이 더 정확합니다.
"""

from train_pretext import train_pretext
from train import train
from evaluate import evaluate


print(f"Start training pretext model...")
pretext_model = train_pretext()
print(f"Finished training pretext model {pretext_model}")

print(f"Start training model...")
model_weight = train(pretext_model)
print(f"Finished training model {model_weight}")

print("Start prediction...")
evaluate(model_weight)
print("Finished!")
