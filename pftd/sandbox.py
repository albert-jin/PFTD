"""Separate candidate and reference processes; no checker assets mounted in Docker."""
import json
import os
import subprocess
import tempfile
import uuid


class InfrastructureError(RuntimeError):
    pass


class Sandbox:
    def __init__(self, image='pftd-checker:local'):
        self.image = image

    def run(self, payload):
        name = 'pftd-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '-i', '--name', name,
                   '--network=none', '--read-only', '--cap-drop=ALL',
                   '--security-opt=no-new-privileges', '--pids-limit=64',
                   '--memory=512m', '--memory-swap=512m', '--cpus=1',
                   '--user=65534:65534', '--tmpfs=/tmp:rw,noexec,nosuid,size=16m',
                   self.image]
        env = os.environ.copy()
        timeout = 45  # container startup; worker enforces the 2-second call limit
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            with tempfile.TemporaryDirectory(prefix='pftd-call-') as cwd:
                result = subprocess.run(command, input=json.dumps(payload), text=True, encoding='utf-8', capture_output=True, cwd=cwd, env=env, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=15)
            raise InfrastructureError('container did not return after worker deadline') from exc
        except OSError as exc:
            raise InfrastructureError('sandbox runtime unavailable') from exc
        if result.returncode == 137:
            return {'ok': False, 'error_type': 'MemoryError', 'error': 'memory cap exceeded'}
        if result.returncode:
            raise InfrastructureError('sandbox exited unexpectedly: ' + result.stderr[-1000:])
        try:
            record = json.loads(result.stdout)
        except ValueError as exc:
            raise InfrastructureError('invalid sandbox response') from exc
        if not isinstance(record, dict) or type(record.get('ok')) is not bool:
            raise InfrastructureError('malformed sandbox result')
        return record
