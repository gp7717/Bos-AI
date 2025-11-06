def rooter():
    try:
        while True:
            n = yield
            result = int(n ** 0.5)
            yield result
    except GeneratorExit:
        return


def squarer():
    try:
        while True:
            n = yield
            result = n * n
            yield result
    except GeneratorExit:
        return


def accumulator():
    total = 0
    try:
        while True:
            n = yield
            total += n
            yield total
    except GeneratorExit:
        return


def pipeline(prod, workers, cons):
    prods = [w() for w in prod]
    works = [w() for w in workers]
    conss = [w() for w in cons]

    for p in prods + works + conss:
        next(p)

    q = prods

    for w in works:
        q.append(w)

    q.extend(conss)

    x = None
    try:
        while True:
            for p in q:
                x = p.send(x)
    except StopIteration:
        pass