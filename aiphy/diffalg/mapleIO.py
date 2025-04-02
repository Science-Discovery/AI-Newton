import subprocess
import os
import signal


class mapleIO:
    def __init__(self):
        self.commands = []
        self.outputs = []

    def import_lib(self, libname: str):
        """
        Some interesting packages:
          ['DifferentialAlgebra', 'DEtools', 'PDEtools', 'Ore_algebra', 'Physics']
        After import DifferentialAlgebra, we can import Tools to use more functions
        """
        self.commands.append(f'with ({libname})')

    def append_command(self, command: str):
        assert '\n' not in command
        self.commands.append(command)

    def exec_maple(self, timeout: int = 60, debug: bool = False):
        exec_cmd = '/opt/maple2024/bin/maple'
        func_cmd = ';\n'.join(self.commands)
        exec_args = f'interface(prettyprint=0):\n{func_cmd};'
        total_times = 0

        while True:
            process = subprocess.Popen(exec_cmd,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       encoding='utf-8',
                                       start_new_session=True,)
            try:
                stdout_, stderr_ = process.communicate(exec_args, timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # If SIGTERM fails, send SIGKILL
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=5)
                stderr_ = f"subprocess.TimeoutExpired: Command '/opt/maple2024/bin/maple' timed out after {timeout} seconds"
            if stderr_:
                if debug:
                    print('-'*50)
                    for c in self.commands:
                        print(c + ';')
                    print('-'*50, '\n', stderr_)
                total_times += 1
                if total_times >= 1:
                    raise Exception(stderr_)
            else:
                break
        self.outputs = self.translate(stdout_)
        for message in self.outputs:
            if 'error' in message.lower():
                if debug:
                    print('debug in maple' + '-'*20)
                    for c in self.commands:
                        print(c + ';')
                    print('debug in maple' + '-'*20)
                raise Exception(message)
        return self.outputs

    def translate(self, stdout: str):
        lines = stdout.split('\n> ')[2:-1]
        outputs = []
        for i, c in enumerate(self.commands):
            temp = lines[i].split(';\n')
            assert temp[0] == c
            assert len(temp) == 2
            outputs.append(temp[1].replace('\n', ' ').replace('\\', ' ').replace(' ', ''))
        return outputs
