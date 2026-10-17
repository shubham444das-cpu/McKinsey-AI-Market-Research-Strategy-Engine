"""
Test each search provider on its own, so you can see exactly which one is failing.

    python check_search.py                      # default query
    python check_search.py "stripe embedded finance 2026"
"""
import sys
import time

from utils import search

query = " ".join(sys.argv[1:]) or "enterprise AI agent adoption 2026"
print(f"Query: {query!r}\n")

ok_any = False
for name, fn in search._PROVIDERS.items():
    t0 = time.time()
    try:
        results = fn(query, 4)
        dt = time.time() - t0
        if results:
            ok_any = True
            print(f"[ OK ] {name:<13} {len(results)} result(s) in {dt:.1f}s")
            for r in results:
                print(f"         - {r['title'][:70]}  {r['url'][:90]}")
        else:
            print(f"[EMPTY] {name:<13} returned 0 results in {dt:.1f}s")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name:<13} {type(e).__name__}: {str(e)[:300]}")
    print()

print("RESULT:", "at least one provider works - the pipeline will find sources."
      if ok_any else "NO provider works - see the errors above (key, quota, network/proxy).")
