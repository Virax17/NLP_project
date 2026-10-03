import sys, json
sys.path.insert(0, ".")
from src.chat_engine import Engine

eng = Engine()

def convo(*msgs):
    ctx = None
    for m in msgs:
        r = eng.respond(m, ctx)
        ctx = r["context"]
        print(f"\nUSER: {m}")
        for rep in r["replies"]:
            t = rep["type"]
            if t == "result":
                print(f"  [result] {rep['disease']['name']} | {rep['confidence']['label']} | alts={[a['name'] for a in rep['alternatives']]}")
            elif t == "info":
                print(f"  [info] {rep['heading']}: {rep['text'][:90]}")
            else:
                print(f"  [{t}] {rep['text'][:110]}")
        print(f"  quick={r['quick']}")
    return ctx

if __name__ == "__main__":
    convo("hi")
    convo("my skin is itchy and red", "elbows", "yes", "no", "yes")
    convo("i have a rash")
    convo("is eczema contagious?")
    convo("what is the difference between eczema and psoriasis")
    convo("itchy rash between my fingers worse at night")
    convo("what's the weather today")
    convo("my head hurts")
    convo("can't breathe and my lips are swollen with hives")
    convo("small pimpls on my face", "How is it treated?")
    convo("thanks", "bye")
