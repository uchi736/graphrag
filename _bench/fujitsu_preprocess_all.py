#!/usr/bin/env python
"""Fujitsu 34PDF を preprocessing_optimizer で markdown化

各PDF → _bench/_pp/{pdf_name}/extracted_text.txt
PaddleX(8005)が落ちてるため backend='none'。テキストPDFはmarkdown構造化、
画像PDFはマーカーのみ。後で fujitsu_ingest_md.py が page単位で再chunkする。

Usage:
    python _bench/fujitsu_preprocess_all.py
    python _bench/fujitsu_preprocess_all.py --skip-existing
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '..' / 'preprocessing_optimizer'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf-dir', default='../Fujitsu-RAG-Hard-Benchmark/dataset/PDFs')
    ap.add_argument('--out-dir', default='_bench/_pp')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    proj = Path(__file__).resolve().parents[1]
    pdf_dir = (proj / args.pdf_dir).resolve()
    out_dir = (proj / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from core.processor import UnifiedProcessor

    pdfs = sorted([f for f in pdf_dir.iterdir() if f.suffix.lower() == '.pdf'])
    print(f'Found {len(pdfs)} PDFs')

    proc = UnifiedProcessor(output_format='text', backend='none')

    t_all = time.time()
    for i, pdf in enumerate(pdfs, 1):
        target = out_dir / pdf.stem
        if args.skip_existing and (target / 'extracted_text.txt').exists():
            sz = (target / 'extracted_text.txt').stat().st_size
            print(f'  [{i:2d}/{len(pdfs)}] skip: {pdf.name} ({sz} bytes)')
            continue
        t = time.time()
        try:
            proc.process(str(pdf), output_dir=str(target), parallel=True)
        except Exception as e:
            # PermissionError on temp_images cleanup は許容
            if 'PermissionError' not in str(type(e).__name__):
                print(f'  ⚠️ {pdf.name}: {e}')
        elapsed = time.time() - t
        ok = (target / 'extracted_text.txt').exists()
        size = (target / 'extracted_text.txt').stat().st_size if ok else 0
        print(f'  [{i:2d}/{len(pdfs)}] {pdf.name}: {elapsed:.0f}s, {size} bytes {"✅" if ok else "❌"}')

    print(f'\n総時間: {time.time()-t_all:.0f}s ({(time.time()-t_all)/60:.1f}min)')


if __name__ == '__main__':
    main()
