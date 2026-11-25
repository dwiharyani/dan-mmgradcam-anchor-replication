#!/usr/bin/env python3
"""Execute a Jupyter notebook top to bottom and save it with its outputs.

Use this only where Jupyter is not installed. With Jupyter available, the equivalent is
    jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb

Each code cell runs in one shared namespace, in order, with the notebook folder as the
working directory. Printed text, the value of the last expression, display() calls and
open matplotlib figures are stored as standard nbformat-4 outputs. Execution stops at the
first error and the error is written into the notebook.

Usage: python tools/execute_notebook.py notebooks/01_*.ipynb [more notebooks]
"""
import ast, base64, io, json, os, sys, time, traceback, types
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Recorder:
    def __init__(self):
        self.outputs, self.buf = [], []

    def write(self, s):
        self.buf.append(s)

    def flush(self):
        pass

    def push_text(self):
        t = ''.join(self.buf)
        self.buf = []
        if t:
            if self.outputs and self.outputs[-1].get('output_type') == 'stream':
                self.outputs[-1]['text'] += t
            else:
                self.outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': t})

    def add(self, data, kind='display_data', count=None):
        self.push_text()
        o = {'output_type': kind, 'data': data, 'metadata': {}}
        if kind == 'execute_result':
            o['execution_count'] = count
        self.outputs.append(o)


REC = None


def mime(obj):
    if hasattr(obj, '_png_bytes'):
        return {'image/png': base64.b64encode(obj._png_bytes).decode(), 'text/plain': '<IPython.core.display.Image object>'}
    d = {'text/plain': repr(obj)}
    if hasattr(obj, '_repr_html_'):
        try:
            h = obj._repr_html_()
            if h:
                d['text/html'] = h
        except Exception:
            pass
    if hasattr(obj, '_repr_markdown_'):
        d['text/markdown'] = obj._repr_markdown_()
    return d


def display(*objs, **kw):
    for o in objs:
        REC.add(mime(o))


class Image:
    def __init__(self, filename=None, data=None, width=None, **kw):
        self._png_bytes = Path(filename).read_bytes() if filename else data


class Markdown:
    def __init__(self, text):
        self.text = text

    def _repr_markdown_(self):
        return self.text

    def __repr__(self):
        return self.text


def install_fake_ipython():
    if 'IPython' in sys.modules:
        return
    ip = types.ModuleType('IPython'); dm = types.ModuleType('IPython.display')
    dm.display, dm.Image, dm.Markdown = display, Image, Markdown
    ip.display = dm
    ip.get_ipython = lambda: None
    ip.version_info = (8, 30, 0)
    sys.modules['IPython'], sys.modules['IPython.display'] = ip, dm


def flush_figures():
    for num in plt.get_fignums():
        fig = plt.figure(num); b = io.BytesIO()
        fig.savefig(b, format='png', dpi=110, bbox_inches='tight')
        REC.add({'image/png': base64.b64encode(b.getvalue()).decode(), 'text/plain': f'<Figure {num}>'})
    plt.close('all')


def run(path):
    global REC
    path = Path(path).resolve()
    nb = json.loads(path.read_text())
    install_fake_ipython()
    ns = {'__name__': '__main__', 'display': display}
    old_cwd, old_out = os.getcwd(), sys.stdout
    os.chdir(path.parent)
    sys.path.insert(0, str(path.parent))
    count, t0, ok = 0, time.time(), True
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        cell['execution_count'] = count
        REC = Recorder(); sys.stdout = REC
        src = ''.join(cell['source'])
        try:
            tree = ast.parse(src)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, f'<cell {count}>', 'exec'), ns)
            val = eval(compile(ast.Expression(last.value), f'<cell {count}>', 'eval'), ns) if last else None
            flush_figures()
            if val is not None:
                REC.add(mime(val), 'execute_result', count)
            REC.push_text()
        except Exception as e:
            REC.push_text()
            REC.outputs.append({'output_type': 'error', 'ename': type(e).__name__, 'evalue': str(e),
                                'traceback': traceback.format_exc().splitlines()})
            ok = False
        finally:
            sys.stdout = old_out
        cell['outputs'] = REC.outputs
        if not ok:
            print(f'ERROR in {path.name} cell {count}:\n' + '\n'.join(REC.outputs[-1]['traceback'][-6:]))
            break
    os.chdir(old_cwd); sys.path.pop(0)
    nb.setdefault('metadata', {})
    nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb['metadata']['language_info'] = {'name': 'python', 'version': sys.version.split()[0]}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    print(f'{path.name}: {count} code cells, {"OK" if ok else "FAILED"}, {time.time() - t0:.0f}s')
    return ok


if __name__ == '__main__':
    results = [run(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
