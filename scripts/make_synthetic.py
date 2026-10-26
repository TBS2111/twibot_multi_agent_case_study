"""Create a small TwiBot-20-format dataset for smoke-testing (NOT real data)."""
import json, random, sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
out.mkdir(exist_ok=True)
random.seed(0)
words = [f"w{i}" for i in range(400)]
N_H, GROUPS, PER = 300, 6, 25
users = []
for g in range(GROUPS):                                   # botnets: shared template/url/hashtag + mutual follows
    ids = [f"u{10000 + g * 100 + i}" for i in range(PER)]
    topic = random.sample(words, 12)
    for i, uid in enumerate(ids):
        tw = [f"{' '.join(random.sample(topic, 6))} #promo{g} http://t.co/bot{g}x{random.randint(0, 3)}" for _ in range(25)]
        fol = random.sample([x for x in ids if x != uid], 8)
        users.append(dict(ID=uid, label="1", tweet=tw, neighbor=dict(following=fol, follower=fol[:4]),
                          profile=dict(followers_count=str(random.randint(0, 50)), friends_count=str(random.randint(500, 3000)),
                                       statuses_count=str(random.randint(2000, 9000)), verified="False ", default_profile="True ",
                                       description="", screen_name=f"bot{g}_{i}", created_at="Tue Mar 03 10:00:00 +0000 2020")))
hum = [f"u{50000 + i}" for i in range(N_H)]
for i, uid in enumerate(hum):
    tw = [" ".join(random.sample(words, random.randint(5, 14))) + (f" #tag{random.randint(0, 80)}" if random.random() < .3 else "")
          for _ in range(random.randint(5, 25))]
    users.append(dict(ID=uid, label="0", tweet=tw,
                      neighbor=dict(following=random.sample(hum, 5), follower=random.sample(hum, 3)),
                      profile=dict(followers_count=str(random.randint(20, 3000)), friends_count=str(random.randint(20, 1500)),
                                   statuses_count=str(random.randint(50, 6000)), verified="False ", default_profile="False ",
                                   description="hello world human", screen_name=f"human{i}", created_at="Tue Mar 03 10:00:00 +0000 2012")))
random.shuffle(users)
n = len(users)
for name, sl in (("train", users[:int(.7 * n)]), ("dev", users[int(.7 * n):int(.85 * n)]), ("test", users[int(.85 * n):])):
    json.dump(sl, open(out / f"{name}.json", "w"))
json.dump([], open(out / "support.json", "w"))
print("wrote synthetic data to", out, "users:", n)
