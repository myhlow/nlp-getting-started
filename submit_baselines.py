import subprocess
import sys
import time

submissions = [
    {
        'file': '/root/nlp-getting-started/submission_majority.csv',
        'message': 'Pocket Data Science V - Exp 01a: Constant 0 (All Zeros)'
    },
    {
        'file': '/root/nlp-getting-started/submission_constant_1.csv',
        'message': 'Pocket Data Science V - Exp 01b: Constant 1 (All Ones)'
    },
    {
        'file': '/root/nlp-getting-started/submission_random.csv',
        'message': 'Pocket Data Science V - Exp 01c: Empirical Random Prior (p=0.430)'
    }
]

def main():
    print("=" * 65)
    print("Submitting Baseline Models to Kaggle Leaderboard (nlp-getting-started)")
    print("=" * 65)

    for item in submissions:
        cmd = [
            '/root/.local/bin/kaggle', 'competitions', 'submit',
            '-c', 'nlp-getting-started',
            '-f', item['file'],
            '-m', item['message']
        ]
        print(f"\nSubmitting {item['file']}...")
        print(f"Message: {item['message']}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", res.stdout.strip())
        if res.stderr.strip():
            print("STDERR:", res.stderr.strip())
        if res.returncode != 0:
            print(f"[!] Submission returned error code {res.returncode}")
        else:
            print("[+] Successfully submitted!")
        time.sleep(2)

if __name__ == '__main__':
    main()
