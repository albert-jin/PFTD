"""PFTD addition: use the new core loop with this pinned release's CodeAct agent.

Install the root PFTD package first; CLI arguments are identical to `pftd train`.
The original SFT and evaluation entry points are retained without changes.
"""
import sys
from pftd.cli import main

if __name__ == '__main__':
    sys.argv.insert(1, 'train')
    main()
