#!/usr/bin/env python3
"""Выгрузка справочников МКБ-10 с портала НСИ Минздрава (ЕГИСЗ).

Тянет два справочника, нужных нормализатору диагнозов (вариант А):

  * Том 1 — МКБ-10 (код, название, иерархия)        OID ...11.1005
  * Том 3 — алфавитный указатель формулировок        OID ...11.1489

Для доступа нужен бесплатный userKey: зарегистрироваться на
https://nsi.rosminzdrav.ru, личный кабинет -> "Ключ доступа к API".
Передать через переменную окружения NSI_USER_KEY.

Примеры:
    # быстрый сэмпл (первая страница каждого справочника) в stdout
    NSI_USER_KEY=xxxx python scripts/fetch_mkb_nsi.py --sample

    # полная выгрузка в data/*.jsonl
    NSI_USER_KEY=xxxx python scripts/fetch_mkb_nsi.py --all --out data
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://nsi.rosminzdrav.ru/port/rest"

# Контекст TLS; --insecure подменяет на неверифицирующий (корп. прокси и т.п.).
_SSL_CTX: ssl.SSLContext | None = None

REFBOOKS = {
    "mkb10_vol1": {
        "oid": "1.2.643.5.1.13.13.11.1005",
        "title": "МКБ-10, Том 1 (справочник)",
    },
    "mkb10_vol3_index": {
        "oid": "1.2.643.5.1.13.13.11.1489",
        "title": "МКБ-10, Том 3 (алфавитный указатель формулировок)",
    },
}

PAGE_SIZE = 100  # максимум, который отдаёт /data за один запрос


def _get(path: str, params: dict[str, str], key: str) -> dict:
    """GET к REST API НСИ с userKey, с парой ретраев на сетевые сбои."""
    query = urllib.parse.urlencode({**params, "userKey": key})
    url = f"{API_BASE}/{path}?{query}"
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 4xx чинить ретраями бессмысленно — отдаём сразу.
            body = e.read().decode("utf-8", "replace")[:300]
            raise SystemExit(f"HTTP {e.code} на {path}: {body}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"Не удалось получить {path}: {last_err}")


def latest_version(oid: str, key: str) -> str:
    """Дата последней версии справочника (НСИ требует version в /data)."""
    data = _get("versions", {"identifier": oid}, key)
    versions = data.get("list") or data.get("versions") or []
    if not versions:
        raise SystemExit(f"У справочника {oid} не нашлось версий: {data}")
    # Берём максимальную дату публикации.
    def ver_key(v: dict) -> str:
        return str(v.get("createDate") or v.get("version") or "")
    return str(max(versions, key=ver_key)["version"])


def _row_to_dict(row: object) -> dict:
    """Запись НСИ -> плоский dict {колонка: значение}.

    /data отдаёт каждую запись списком ячеек {"column": ..., "value": ...}.
    На всякий случай поддерживаем и вариант, где запись уже dict.
    """
    if isinstance(row, dict):
        return row
    out: dict[str, object] = {}
    if isinstance(row, list):
        for cell in row:
            if isinstance(cell, dict) and "column" in cell:
                out[cell["column"]] = cell.get("value")
    return out


def iter_records(oid: str, version: str, key: str, max_rows: int | None):
    """Постранично выгружает записи справочника."""
    page = 1
    fetched = 0
    while True:
        data = _get(
            "data",
            {
                "identifier": oid,
                "version": version,
                "page": str(page),
                "size": str(PAGE_SIZE),
            },
            key,
        )
        rows = data.get("list") or []
        if not rows:
            break
        for row in rows:
            yield _row_to_dict(row)
            fetched += 1
            if max_rows is not None and fetched >= max_rows:
                return
        total = int(data.get("total") or 0)
        if total and fetched >= total:
            break
        page += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Выгрузка МКБ-10 с НСИ Минздрава")
    parser.add_argument("--sample", action="store_true",
                        help="только сэмпл (15 записей) в stdout, без записи файлов")
    parser.add_argument("--all", action="store_true",
                        help="полная выгрузка обоих справочников")
    parser.add_argument("--out", default="data",
                        help="каталог для JSONL (по умолчанию ./data)")
    parser.add_argument("--limit", type=int, default=15,
                        help="сколько записей показать в режиме --sample")
    parser.add_argument("--insecure", action="store_true",
                        help="не проверять TLS-сертификат (корп. прокси/MITM)")
    args = parser.parse_args()

    if args.insecure:
        global _SSL_CTX
        _SSL_CTX = ssl._create_unverified_context()
        print("ВНИМАНИЕ: проверка TLS-сертификата отключена (--insecure)",
              file=sys.stderr)

    key = os.environ.get("NSI_USER_KEY")
    if not key:
        print("Нужен ключ: NSI_USER_KEY=... (личный кабинет на nsi.rosminzdrav.ru)",
              file=sys.stderr)
        return 2
    if not (args.sample or args.all):
        args.sample = True  # дефолт — посмотреть сэмпл

    if args.all:
        os.makedirs(args.out, exist_ok=True)

    for name, ref in REFBOOKS.items():
        oid, title = ref["oid"], ref["title"]
        version = latest_version(oid, key)
        print(f"\n=== {title}\n    OID {oid}, версия {version}", file=sys.stderr)

        if args.all:
            path = os.path.join(args.out, f"{name}.jsonl")
            count = 0
            with open(path, "w", encoding="utf-8") as f:
                for rec in iter_records(oid, version, key, max_rows=None):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 1000 == 0:
                        print(f"    ...{count}", file=sys.stderr)
            # meta-файл рядом — источник версии для provenance (ClassifierRef).
            meta_path = os.path.join(args.out, f"{name}.meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"oid": oid, "version": version, "title": title, "count": count},
                    f,
                    ensure_ascii=False,
                )
            print(f"    записано {count} -> {path} (версия {version})", file=sys.stderr)
        else:
            for rec in iter_records(oid, version, key, max_rows=args.limit):
                print(json.dumps(rec, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
