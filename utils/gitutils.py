import subprocess


def save_git_info(diff_file):
    with open(diff_file, 'wb') as f:
        f.write(bytes("\n**************   Git Branch Information   **********:\n", 'utf-8'))
        cmd = 'git branch -vv'
        log_cmd(cmd, f)

        f.write(bytes("\n**************   Git Log Information   **********:\n", 'utf-8'))
        # cmd = 'git log --oneline --graph'.split()
        cmd = 'git log --oneline --graph | head -20'
        log_cmd(cmd, f)

        f.write(bytes("\n**************   Git Diff with HEAD   **********:\n", 'utf-8'))
        cmd = 'git diff HEAD'
        log_cmd(cmd, f)


def log_cmd(cmd, f):
    out = subprocess.run(['-c', cmd], shell=True, capture_output=True)
    f.write(out.stdout)
