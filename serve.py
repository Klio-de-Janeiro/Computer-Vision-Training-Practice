"""Local-only server. No third-party packages, notebook execution or external APIs."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / 'dist'
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    except OSError as exc:
        print(f'Cannot start on port {args.port}: {exc}. Try --port 8766.')
        raise SystemExit(1)
    url = f'http://127.0.0.1:{args.port}'
    print(f'CV exam: {url} (Ctrl+C to stop)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
