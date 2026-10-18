"""Rebuild an offline, read-only snapshot of the course notebooks. Standard library only."""
import base64
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
COURSE = APP.parents[1] / 'cv_course_at_mipt'
DIST = APP / 'dist'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', type=Path, default=COURSE, help='Path to the local course repository')
    args = parser.parse_args()
    course = args.course.resolve()
    if not course.is_dir() or not any(course.glob('lesson_*/*.ipynb')):
        raise SystemExit('Course notebooks not found. Use --course PATH_TO_COURSE.')
    assets = DIST / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    private = DIST / 'local'
    private.mkdir(parents=True, exist_ok=True)
    notebooks = []
    for path in sorted(course.glob('lesson_*/**/*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        rel = path.relative_to(course).as_posix()
        ident = hashlib.sha256(rel.encode()).hexdigest()[:12]
        cells = []
        for index, cell in enumerate(nb['cells'], 1):
            source = ''.join(cell.get('source', []))
            for name, formats in cell.get('attachments', {}).items():
                for mime, ext in [('image/png', 'png'), ('image/jpeg', 'jpg')]:
                    if mime in formats:
                        target = f'{ident}-{index}-{hashlib.sha256(name.encode()).hexdigest()[:8]}.{ext}'
                        (assets / target).write_bytes(base64.b64decode(''.join(formats[mime])))
                        source = source.replace('attachment:' + name, 'assets/' + target)
                        break
            def local_image(match):
                label, url = match.groups()
                if url.startswith(('http:', 'https:', 'data:', 'assets/' + ident)):
                    # External images are not requested at runtime.
                    return f'[{label or "Внешняя иллюстрация"}]({url})' if url.startswith('http') else match.group()
                src = (path.parent / url.split('#')[0]).resolve()
                if src.is_relative_to(course) and src.is_file():
                    target = hashlib.sha256(str(src).encode()).hexdigest()[:16] + src.suffix
                    shutil.copyfile(src, assets / target)
                    return f'![{label}](assets/{target})'
                return f'*Иллюстрация недоступна в исходных файлах: {label}*'
            if cell['cell_type'] == 'markdown':
                source = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', local_image, source)
            images = []
            for out in cell.get('outputs', []):
                data = out.get('data', {})
                if 'image/png' in data:
                    target = f'{ident}-{index}-out{len(images)}.png'
                    (assets / target).write_bytes(base64.b64decode(''.join(data['image/png'])))
                    images.append('assets/' + target)
            if source.strip():
                cells.append({'number': index, 'type': cell['cell_type'], 'text': source, 'images': images})
        notebooks.append({'id': ident, 'lesson': int(path.parts[len(course.parts)].split('_')[1]),
                          'path': rel, 'title': path.stem, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cells': cells})
    manifest = []
    for notebook in notebooks:
        ident = notebook['id']
        payload = 'window.CV_NOTEBOOKS = window.CV_NOTEBOOKS || {};\nwindow.CV_NOTEBOOKS[' + json.dumps(ident) + '] = ' + json.dumps(notebook, ensure_ascii=False) + ';\n'
        (private / f'{ident}.js').write_text(payload, encoding='utf-8')
        manifest.append({k:v for k,v in notebook.items() if k!='cells'} | {
            'cellCount':len(notebook['cells']), 'imageCount':sum(len(c['images']) for c in notebook['cells'])})
    # Public metadata is safe to track; notebook bodies and images stay ignored locally.
    (DIST / 'sources.js').write_text('window.CV_SOURCES = ' + json.dumps(manifest, ensure_ascii=False) + ';\n', encoding='utf-8')
    print(f'{len(notebooks)} notebooks, {sum(len(n["cells"]) for n in notebooks)} cells, {len(list(assets.iterdir()))} local images')

if __name__ == '__main__':
    main()
