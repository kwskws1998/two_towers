"""
vast.ai 셋업 스크립트 - 한 번만 실행하면 됨
사용법: python setup_et_models.py --et2-checkpoint ./checkpoints/et_predictor2_seed123

ET model 1 (Huang & Hollenstein 2023):
  - SelectiveCacheForLM 레포에서 가중치 받아서 eyetrackpy 패키지 내부에 복사

ET model 2 (Li & Rudzicz 2021):
  - 로컬 체크포인트가 없으면 HF(repo: skboy/et_prediction_2)에서 자동 다운로드
  - 체크포인트(.pt / .safetensors) 경로를 환경변수로 등록
"""

import argparse
import os
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen


def run(cmd, check=True):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def find_eyetrackpy_root():
    try:
        import eyetrackpy
        return os.path.dirname(eyetrackpy.__file__)
    except ImportError:
        return None


def install_packages():
    print("\n[1/4] eyetrackpy, tokenizer_aligner 설치 중...")
    run("pip install git+https://github.com/angelalopezcardona/tokenizer_aligner.git@v1.0.0 -q")
    run("pip install git+https://github.com/angelalopezcardona/eyetrackpy.git@v1.0.0 -q")
    run("pip install safetensors -q")


def setup_et_model1(clone_dir="./SelectiveCacheForLM"):
    print("\n[2/4] ET model 1 가중치 설정 중 (Huang & Hollenstein 2023)...")

    # SelectiveCacheForLM 클론
    if not os.path.isdir(clone_dir):
        run(f"git clone https://github.com/huangxt39/SelectiveCacheForLM.git {clone_dir}")
    else:
        print(f"  {clone_dir} 이미 존재, 스킵")

    src = os.path.join(clone_dir, "FPmodels", "T5-tokenizer-BiLSTM-TRT-12-concat-3")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"가중치 파일을 찾을 수 없습니다: {src}\n"
            "SelectiveCacheForLM 레포 구조가 바뀌었을 수 있습니다. 직접 확인해주세요."
        )

    # eyetrackpy 패키지 내부 경로에 복사
    et_root = find_eyetrackpy_root()
    if et_root is None:
        raise ImportError("eyetrackpy가 설치되지 않았습니다. install_packages()를 먼저 실행하세요.")

    dst_dir = os.path.join(
        et_root, "data_generator", "fixations_predictor_trained_1"
    )
    dst = os.path.join(dst_dir, "T5-tokenizer-BiLSTM-TRT-12-concat-3")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isfile(dst):
        shutil.copy2(src, dst)
        print(f"  가중치 복사 완료: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
    else:
        print(f"  이미 존재: {dst}")


def setup_et_model2(checkpoint_path):
    print("\n[3/4] ET model 2 체크포인트 확인 중 (Li & Rudzicz 2021)...")

    # .pt 또는 .safetensors 중 존재하는 것 탐색
    resolved = None
    for ext in ["", ".safetensors", ".pt", ".bin"]:
        candidate = checkpoint_path + ext if not checkpoint_path.endswith(ext) else checkpoint_path
        if os.path.isfile(candidate):
            resolved = candidate
            break

    return resolved


def _download_et2_checkpoint_from_hf(destination_path, repo_id, filename):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    print(f"  로컬 체크포인트가 없어 HF에서 다운로드 시도: {url}")

    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=120) as response, open(destination_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    return destination_path


def resolve_or_download_et_model2(
    checkpoint_path,
    auto_download=True,
    hf_repo_id="skboy/et_prediction_2",
    hf_filename="et_predictor2_seed123.safetensors",
):
    resolved = setup_et_model2(checkpoint_path)

    if resolved is None and auto_download:
        # 확장자 없는 기본 경로면 .safetensors로 저장
        if checkpoint_path.endswith((".safetensors", ".pt", ".bin")):
            destination = checkpoint_path
        else:
            destination = checkpoint_path + ".safetensors"
        try:
            _download_et2_checkpoint_from_hf(destination, hf_repo_id, hf_filename)
        except Exception as exc:
            raise FileNotFoundError(
                f"ET model 2 체크포인트를 찾을 수 없습니다: {checkpoint_path}[.pt/.safetensors]\n"
                f"그리고 HF 자동 다운로드도 실패했습니다 ({hf_repo_id}/{hf_filename}).\n"
                "네트워크 또는 파일명을 확인해주세요."
            ) from exc
        resolved = setup_et_model2(checkpoint_path)

    if resolved is None:
        raise FileNotFoundError(
            f"ET model 2 체크포인트를 찾을 수 없습니다: {checkpoint_path}[.pt/.safetensors]\n"
            "노트북에서 학습한 checkpoints/et_predictor2_seed123.pt (또는 .safetensors)를 "
            "지정하거나, --et2-hf-repo/--et2-hf-filename 옵션을 사용하세요."
        )

    print(f"  체크포인트 확인: {resolved} ({os.path.getsize(resolved)/1e6:.1f} MB)")

    # 환경변수 등록 안내 + .env 파일 생성
    abs_path = os.path.abspath(resolved)
    env_line = f"ET2_CHECKPOINT_PATH={abs_path}"

    with open(".env_et", "w") as f:
        f.write(env_line + "\n")

    print(f"\n  아래 명령을 실행하거나 .env_et를 source 해주세요:")
    print(f"  export {env_line}")
    print(f"  또는: source .env_et")

    # 현재 프로세스에도 등록
    os.environ["ET2_CHECKPOINT_PATH"] = abs_path
    return abs_path


def verify_setup():
    print("\n[4/4] 설치 검증 중...")

    # eyetrackpy model 1
    try:
        from eyetrackpy.data_generator.fixations_predictor_trained_1.fixations_predictor_model_1 import FixationsPredictor_1
        print("  ✓ FixationsPredictor_1 import OK")
    except Exception as e:
        print(f"  ✗ FixationsPredictor_1 import 실패: {e}")

    # et2_wrapper
    try:
        from va_gaze.models.et2_wrapper import FixationsPredictor_2
        print("  ✓ FixationsPredictor_2 (wrapper) import OK")
    except Exception as e:
        print(f"  ✗ FixationsPredictor_2 wrapper import 실패: {e}")
        print("    et2_wrapper.py가 같은 디렉토리에 있는지 확인하세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--et2-checkpoint",
        default="./checkpoints/et_predictor2_seed123",
        help="ET model 2 체크포인트 경로 (확장자 없어도 됨, .pt/.safetensors 자동 탐색)",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="pip install 생략 (이미 설치된 경우)"
    )
    parser.add_argument(
        "--clone-dir", default="./SelectiveCacheForLM",
        help="SelectiveCacheForLM 클론 경로"
    )
    parser.add_argument(
        "--skip-et1",
        action="store_true",
        help="ET model 1(SelectiveCacheForLM) 설정을 건너뜀",
    )
    parser.add_argument(
        "--no-et2-auto-download",
        action="store_true",
        help="로컬 ET2 체크포인트가 없을 때 HF 자동 다운로드를 비활성화",
    )
    parser.add_argument(
        "--et2-hf-repo",
        default="skboy/et_prediction_2",
        help="ET2 자동 다운로드에 사용할 Hugging Face repo id",
    )
    parser.add_argument(
        "--et2-hf-filename",
        default="et_predictor2_seed123.safetensors",
        help="ET2 자동 다운로드에 사용할 Hugging Face 파일명",
    )
    args = parser.parse_args()

    if not args.skip_install:
        install_packages()

    if args.skip_et1:
        print("\n[2/4] ET model 1 설정 건너뜀 (--skip-et1)")
    else:
        setup_et_model1(args.clone_dir)

    resolve_or_download_et_model2(
        args.et2_checkpoint,
        auto_download=not args.no_et2_auto_download,
        hf_repo_id=args.et2_hf_repo,
        hf_filename=args.et2_hf_filename,
    )
    verify_setup()

    print("\n✓ 셋업 완료. 이제 train을 실행하세요.")
    print("  예시: python train_model.py xlmroberta-large mse \\")
    print("          --use-gaze-concat --et2-checkpoint ./checkpoints/et_predictor2_seed123 \\")
    print("          --features-used 1,1,1,1,1 --fp-dropout 0.1,0.3 --batch-size 8 --maxlen 200")


if __name__ == "__main__":
    main()
