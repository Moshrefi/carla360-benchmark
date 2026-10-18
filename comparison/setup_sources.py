"""Download pinned upstream sources without modifying existing checkouts."""
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    sources = json.loads((ROOT/'comparison/sources.json').read_text())
    for name, spec in sources.items():
        dst = ROOT/'third_party'/spec['folder']
        if not dst.exists():
            subprocess.run(['git','clone','--no-checkout',spec['url'],str(dst)],check=True)
            subprocess.run(['git','-C',str(dst),'checkout','--detach',spec['commit']],check=True)
        actual = subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()
        if actual != spec['commit']:
            raise RuntimeError(f'{dst}: expected {spec["commit"]}, found {actual}; existing checkout preserved')
        print(name, actual)
if __name__ == '__main__': main()
