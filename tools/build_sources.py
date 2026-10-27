"""Rebuild an offline, read-only snapshot of the course notebooks. Standard library only."""
import base64
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from html.parser import HTMLParser

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
    paths=list(course.glob('lesson_*/**/*.ipynb'))
    paths += [course/p for p in ['lesson_2/hw/mean-shift.md','lesson_2/hw/seam-carving.md','lesson_5/hw/hw-assignment.md']]
    for path in sorted(paths):
        if int(path.relative_to(course).parts[0].split('_')[1])>7:continue
        document=path.suffix!='.ipynb'
        nb = ({'cells':[{'cell_type':'markdown','source':[path.read_text(encoding='utf-8')]}]} if document else json.loads(path.read_text(encoding='utf-8')))
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
            def copy_image(label, url):
                if url.startswith(('http:', 'https:', 'data:', 'assets/')):
                    # External images are not requested at runtime.
                    return f'[{label or "Внешняя иллюстрация"}]({url})' if url.startswith('http') else f'![{label}]({url})'
                src = (path.parent / url.split('#')[0]).resolve()
                if src.is_relative_to(course) and src.is_file():
                    target = hashlib.sha256(str(src).encode()).hexdigest()[:16] + src.suffix
                    shutil.copyfile(src, assets / target)
                    return f'![{label}](assets/{target})'
                return f'*Иллюстрация недоступна в исходных файлах: {label}*'
            def local_image(match):
                return copy_image(*match.groups())
            def html_image(match):
                class ImageTag(HTMLParser):
                    def handle_starttag(self, tag, attrs):
                        if tag.lower()=='img':self.attrs=dict(attrs)
                parsed=ImageTag();parsed.feed(match.group())
                attrs=getattr(parsed,'attrs',{})
                return copy_image(attrs.get('alt') or 'Иллюстрация',attrs.get('src') or '')
            if cell['cell_type'] == 'markdown':
                source = re.sub(r'<img\b[^>]*>', html_image, source, flags=re.IGNORECASE)
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
                          'path': rel, 'title': path.stem, 'document':document, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cells': cells})
    manifest = []
    for notebook in notebooks:
        ident = notebook['id']
        payload = 'window.CV_NOTEBOOKS = window.CV_NOTEBOOKS || {};\nwindow.CV_NOTEBOOKS[' + json.dumps(ident) + '] = ' + json.dumps(notebook, ensure_ascii=False) + ';\n'
        (private / f'{ident}.js').write_text(payload, encoding='utf-8', newline='\n')
        manifest.append({k:v for k,v in notebook.items() if k!='cells'} | {
            'cellCount':len(notebook['cells']), 'imageCount':sum(len(c['images']) for c in notebook['cells'])})
    # Public metadata is safe to track; notebook bodies and images stay ignored locally.
    (DIST / 'sources.js').write_text('window.CV_SOURCES = ' + json.dumps(manifest, ensure_ascii=False) + ';\n', encoding='utf-8', newline='\n')
    print(f'{len(notebooks)} notebooks, {sum(len(n["cells"]) for n in notebooks)} cells, {len(list(assets.iterdir()))} local images')

if __name__ == '__main__':
    main()
