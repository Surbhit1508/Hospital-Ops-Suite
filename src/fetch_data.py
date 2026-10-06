"""
Fetch real public datasets for the Hospital Operations Intelligence Suite.

Why curl-and-not-requests: Walmart's corporate proxy (sysproxy.wal-mart.com:8080)
does NTLM auth that Python's requests/huggingface_hub libraries can't negotiate,
but curl handles it fine via Windows SSPI. See README.md "Infra Notes" section.

Why the Artifactory URL for HF datasets instead of huggingface.co directly:
Hugging Face's newer "Xet" storage backend (us.aws.cdn.hf.co / xethub.hf.co)
is unreachable through the corporate proxy for any file big enough to be
LFS-backed. Walmart's Artifactory mirrors Hugging Face Hub content and serves
those same files without touching Xet at all.
"""
import subprocess
import sys
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

HF_ARTIFACTORY_BASE = (
    "https://generic.ci.artifacts.walmart.com/artifactory/api/huggingfaceml/"
    "hub-huggingfaceml-release-remote/datasets"
)

PROXY_ENV = {
    "HTTP_PROXY": "http://sysproxy.wal-mart.com:8080",
    "HTTPS_PROXY": "http://sysproxy.wal-mart.com:8080",
}

# (output filename, source URL)
FILES = [
    # Classification target: readmitted (binary). Regression target: time_in_hospital.
    # Real UCI "Diabetes 130-US Hospitals for Years 1999-2008" data, pre-cleaned by imodels.
    ("diabetes_readmission_train.csv", f"{HF_ARTIFACTORY_BASE}/imodels/diabetes-readmission/resolve/main/train.csv"),
    ("diabetes_readmission_test.csv", f"{HF_ARTIFACTORY_BASE}/imodels/diabetes-readmission/resolve/main/test.csv"),
    # NLP: real de-identified medical transcription notes labeled by specialty (MTSamples corpus).
    ("mtsamples_train.parquet", f"{HF_ARTIFACTORY_BASE}/galileo-ai/medical_transcription_40/resolve/main/data/train-00000-of-00001.parquet"),
    ("mtsamples_test.parquet", f"{HF_ARTIFACTORY_BASE}/galileo-ai/medical_transcription_40/resolve/main/data/test-00000-of-00001.parquet"),
]

# Forecasting + optimization data comes straight from the huggingface.co HF Hub mirror of
# HHS/CDC HealthData.gov datasets (small CSVs, not Xet-backed, so plain huggingface.co works).
HHS_FILES = [
    (
        "weekly_covid_hospitalizations.csv",
        "https://huggingface.co/datasets/HHS-Official/weekly-united-states-covid-19-hospitalization-metr/resolve/main/data/dataset.csv",
    ),
    (
        "hospital_capacity_by_state.csv",
        "https://huggingface.co/datasets/HHS-Official/covid-19-reported-patient-impact-and-hospital-capa/resolve/main/data/dataset.csv",
    ),
]


def curl_download(url: str, out_path: Path, timeout: int = 120) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sL", "--max-time", str(timeout), url, "-o", str(out_path)]
    env = {**_base_env(), **PROXY_ENV}
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {result.stderr}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {out_path}")


def _base_env():
    import os

    return dict(os.environ)


def main() -> None:
    all_files = FILES + HHS_FILES
    for filename, url in all_files:
        out_path = DATA_RAW / filename
        if out_path.exists():
            print(f"[skip] {filename} already present ({out_path.stat().st_size:,} bytes)")
            continue
        print(f"[fetch] {filename} <- {url}")
        curl_download(url, out_path)
        print(f"[done] {filename} ({out_path.stat().st_size:,} bytes)")

    print("\nAll datasets fetched into", DATA_RAW)


if __name__ == "__main__":
    sys.exit(main())
