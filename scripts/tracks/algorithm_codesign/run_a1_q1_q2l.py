"""Execute only the authorized non-sampling A1 Q1/Q2-L scope once."""
import argparse
import json

from trottertracks.algorithm_codesign.a1_law_validation import run, diagnose


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="fresh output directory; never overwritten")
    parser.add_argument("--diagnose", action="store_true", help="record source/layout diagnosis without CDF")
    arguments = parser.parse_args()
    result = (diagnose if arguments.diagnose else run)(arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
