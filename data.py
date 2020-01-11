import os

import random
import datetime


def char_range(c1, c2):
    return [chr(c) for c in range(ord(c1), ord(c2) + 1)]


DIGITS = [str(d) for d in list(range(0, 10))]
HEX_DIGITS = DIGITS + char_range('a', 'f')
ALPHA = char_range('a', 'z')
ALPHANUM = DIGITS + ALPHA + char_range('A', 'Z') + list("/-:")


def get_random_string(possible_chars, mu=16, sig=8):
    str_len = max(5, int(random.gauss(mu, sig)))
    return "".join([random.choice(possible_chars) for _ in range(str_len)])


def get_random_id_string():
    r = random.random()
    if r < 0.75:
        id_str = get_random_string(random.choice([DIGITS, HEX_DIGITS, ALPHANUM]))
    else:
        id_str = get_random_string(ALPHA, mu=3, sig=1) + get_random_string(
            random.choice([DIGITS, HEX_DIGITS, ALPHANUM]))

    return id_str


def get_random_date():
    try:
        return datetime.datetime.strptime('{} {}'.format(random.randint(1, 366), random.randint(1950, 2021)), '%j %Y')
    except ValueError:
        get_random_date()


def get_random_date_format(date):
    if random.random() < 0.5:
        sep = random.choice("./-")
        first = random.choice(["%m", "%d"])
        second = "%m" if first == "%d" else "%d"
        year = random.choice(["%Y", "%y"])
        return date.strftime(f"{first}{sep}{second}{sep}{year}")
    else:
        def get_suffix(day):
            return 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        sep = random.choice([",", ", ", " "])

        weekday = "" if random.random() > 0.5 else date.strftime(random.choice(["%a", "%A"]))
        if len(weekday) > 0:
            weekday += sep
            sep = random.choice([",", ", ", " "])

        month = date.strftime(random.choice(["%b", "%B"]))

        suffix = "" if random.random() > 0.5 else get_suffix(date.day)
        day = str(date.day) + suffix

        year = random.choice(["%Y", "%y", "'%y"])
        positional = [day, month]
        random.shuffle(positional)
        format = f"{weekday}{positional[0]} {positional[1]}{sep}{year}"
        return date.strftime(format)


def read_csv(filename):
    with open(filename, 'r') as data:
        return [i.rstrip() for i in data]


def create_splits(train_f, test_f, val_f, train_prop=0.9, val_prop=0.05, gen_n=10000):
    dates = [(0, get_random_date_format(get_random_date())) for _ in range(gen_n)]
    locations = [(1, l) for l in read_csv(os.path.join(DIR, "locations.csv"))]
    randoms = [(2, get_random_id_string()) for _ in range(gen_n)]
    companies = [(3, c) for c in read_csv(os.path.join(DIR, "companies.csv"))]
    goods = [(4, g) for g in read_csv(os.path.join(DIR, "goods.csv"))]

    data = dates + locations + randoms + companies + goods
    random.shuffle(data)
    train_cut = int(train_prop * len(data))
    val_cut = int((train_prop + val_prop) * len(data))

    def write_split(filename, d):
        with open(filename, "w") as split:
            for label, text in d:
                split.write(f"{label}\t{text}\n")

    write_split(train_f, data[:train_cut])
    write_split(val_f, data[train_cut:val_cut])
    write_split(test_f, data[val_cut:])


if __name__ == "__main__":
    DIR = "/Users/fraser/Documents/Personal Projects/vectorai_challenge/data"
    TRAIN = os.path.join(DIR, "train.tsv")
    TEST = os.path.join(DIR, "test.tsv")
    VAL = os.path.join(DIR, "val.tsv")
    create_splits(TRAIN, TEST, VAL)
