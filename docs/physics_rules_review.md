# Movable or fixed: the rules reviewed object by object (2026-11-27)

`opticalnav_sim/physics_objects.py` decides, per scene, which objects a robot can push (PUSH: movable, bottom on the
floor within 0.4 m), which move but are out of its reach (MOVE: on a shelf, a table, a bench) and which are fixed (FIX).
Every object of the 10 scenes that had a `physics.json` (1,935 objects) was looked at and judged by eye, and the rules
were redesigned against those judgements.

## How

1. `tools/physics_review.py` draws contact sheets: per object, a tile with the object in orange in the Solid view of its
   surroundings, the object alone, and its verdict, rule and features (size, mass, support, height over the floor,
   walls touched). Objects with the same label stem, verdict and rule are drawn four at a time (PH40's 659 leaves).
   100 sheets for the 10 scenes.
2. Every sheet was reviewed against written definitions of PUSH / MOVE / FIX (wall-mounted, ceiling and built-in things,
   markings, pieces of larger things: FIX; free things on shelves and tables: MOVE; free things on the floor up to
   ~50 kg: PUSH) and the grouping judged too (one object = one physical thing). Per object: `should`, `grouping`,
   `problem`.
3. `tools/physics_eval.py` scores the rules as the code has them against those judgements, part by part (so a change of
   grouping is scored too) and per object (its parts' majority). UNSURE objects and wrongly grouped ones are left out.

## Results (objects; 1,663 judged with a clear verdict and a sound grouping)

| | agree | PUSH found | MOVE taken for PUSH | PUSH taken for MOVE / FIX |
|---|---|---|---|---|
| before | 94.9 % | 65 / 111 | 22 | 20 / 26 |
| after | 99.5 % | 111 / 111 | 1 | 0 / 0 |

Per scene after: PH33, PH45, ph19, office, shop, warehouse 100 %; PH40 99.7 %; CS02 99.4 %; cafe 98.6 %; apartment 97.7 %.
The rules were tuned on these same scenes: a scene not among them (PH59 and after) is the honest test.

## What was wrong, and the change

| what the review found | count | change |
|---|---|---|
| ground named by its material and merged with buildings (PH45 `ph45_galvanised`, PH40 `ph40_paint_white`); a raised car park (`lot`) over the terrain taken for a table top | 25 | a part with more than 20 m² of upward faces is ground; a flat part wider than 20 m² stays ground unless it is 2.5 m over other ground |
| items on the lowest shelf of a rack or bookcase counted as within reach (bottom < 0.4 m) | 22 | within reach only when standing on the floor |
| fixed by a word: `tree` (potted olive trees), `truck` (hand truck), `platform` (platform cart), `step` (step ladder), `harness` (a lead on a bench), `planter` | 13 | compounds read first (hand truck, platform cart, step ladder, fire extinguisher, floor lamp, lead harness); `tree`, `planter` judged as floor objects when standing free; `harness` out of the fixed words |
| light floor things not "free": floor lamps (1.3-1.5 m), coffee tables (0.9-1.2 m²), potted plants touching a wall | 7 | taller (to 2.2 m) when under 10 kg, wider (to 1.5 m²) when under 0.6 m tall and 20 kg, touching a wall allowed under 10 kg |
| downlights "on" a ceiling duct taken for small items | 3 | carried by structure more than 2 m over the floor: a fixture |
| a tactile floor strip (6 mm) pushed | 1 | on the floor and under 2 cm: part of the floor |
| a 0.03 m gap under a lamp or a pot read as "nothing under it" | 2 | support found up to 5 cm under an object |
| a chair, a 1.9 kg stage "holding up" a wall or its own columns | 2 | holding up only counts for things over 5 kg without a movable word |
| leaves "on" leaves made movable | 3 | a thing without a keyword on another of its own label stem is a piece of it |
| a counter's legs (2.38 x 0.04 m) free-standing | 1 | not free: a slab thinner than 6 cm and longer than 1 m |
| outdoor shrubs pushed once `tree` stopped fixing them | 10 | a plant on the floor without a pot word (pot, planter, container, vase) or in soil is planted: fixed |
| tiny loose things (a 38 g trinket) fixed | 2 | tiny under 30 g (was 50 g) |
| wet-floor sign fixed by `sign` (before the review) | 1 | `sign`, `board`, `banner`, `bollard`, `barrier` on the floor off the walls: judged as a floor object |

## Grouping (one object = one physical thing)

The review marked 213 objects "partial" (a piece of a thing, the rest elsewhere) and 102 "merged" (several things in
one). 17 of them are things that move; the rest are fixed (a wall in two halves, a rig's T-slots), where a split
changes nothing. `tools/physics_eval.py` counts, per scene, the partial objects now joined to something and the joins
of two objects that were sound and apart (a join too many).

| what kept a piece apart | change |
|---|---|
| a piece (a case's carry handle, 8 mm over the case's ribs) was only joined to parts of *another* asset (meant for a keyboard's keys) | a piece also joins a part of its own asset that is no piece |
| a screen's trace (a 3 mm sheet, 19 cm: no "piece" by size) on its display | a sheet under 6 mm and 0.5 m lying on a face (60 % of it touching) is a piece |
| structure words on an asset's small members (a metrology stage's `column`s, a pendant lamp's `dome`) | under 1 m and of an asset: the asset's member, not the building's |
| a .blend object split by material (`stanchions_ph33_black_metal` bases, `stanchions_ph33_steel` posts) | `<object>_<scene code>_<material>`: the object is the asset |
| the stage's `stage` (a theatre's word) | compound "metrology stage" |
| a mouse lying in the ring of a cable coil "contained" in it | a container is at least as tall as what it holds; two things of different movable words stay two |

Partial objects joined: 67 of 213 (PH33 2/2, cafe 14/18, warehouse 3/4, CS02 44/142: the cases with their handles,
the instruments with their traces, the metrology stage with its columns); verdicts unchanged (99.5 %). Joins too many:
two PH40 leaves joined to their branch (right, as a plant), a CS02 ceiling junction box with its conduit saddles (both
fixed).

## Left as it is

- Things merged in one mesh that are not several of one movable thing (ph19's extinguisher plates in a material's mesh,
  a shop gondola with its products, desk items baked into an office pod): a material's mesh says nothing about what its
  pieces are (split, PH45's `ph45_wood_warm` and `ph45_paint_yellow` came apart into "free-standing" benches and wheel
  stops, PH33's sconce mesh into 12 pushable lamps), so it stays whole.

## Several things in one shape (2026-11-28)

PH33's bins are instances of an asset (`PH_metal_trash_can`) modelled as two cans: each instance was one object, and as
an instance (not a mesh) the merge kept it as it was, so the render server never moved it (it moves `dyn_` meshes).
Now:

- `physics_objects.split_pieces`: a shape whose label is a movable word and that falls apart into 2 to 16 pieces (its
  connected parts, vertices welded, joined when their boxes come within 2 cm: a lid stays with its bin), each at least
  0.1 m long and each standing on its own (from the shape's bottom, 80 % of its height: not a pallet's boards, not a
  loaded pallet's layers), is read as one part a piece: `<shape>__p<k>`, k in the order of the pieces' lowest corners.
  An instance's pieces are found once per shape group, in the group's frame (PH45 draws a 287,000-triangle shrub 300
  times), and ordered where the instance puts them.
- `tools/merge_meshes.py`, before the merge: a shape with a piece in a movable object, and any instance of one, is
  written as plain meshes, one per piece and material (`<shape>__p<k>m<j>`, its normals and uvs as Mitsuba loads it,
  the pieces found the same way); those merge into the object's `dyn_` meshes. The manifest's `expanded` records put
  the shape back (`merged_meshes.restore_text`): the scene reads, and unmerges, as it was.

PH33 after: 6 bins (9.6-9.7 kg each), each its own `dyn_` meshes; pushing one in a render moves it with its lid and
leaves its twin. Elsewhere the rule splits only bikes' and hand trucks' wheel pairs and two pallet racks' frames, which
join their objects again.
- Still split, all fixed: Infinigen windows (frame, blind, glass), mirrors (glass, frame), doors (leaf, strip and
  handle), walls in two heights, CS02's rig members.
- Remaining verdict misses: a blanket draped over a bed (large: MOVE expected), a spoon (7 g), two 19 g meter faces, a
  0.62 m task lamp on a bench, a cup at a table's edge, a planter on a high shelf, two leaves on a branch.
- Mass estimates: metal trash cans at 48 kg, a motorcycle at 59 kg (shells measured as solids or the other way round).

## Decisions set by hand

The viewer's selection (click an object, then 고정 / 밂 / 움직임 / 규칙대로) writes `physics_overrides.json` beside
`physics.json` (over the rules on the next tagging, 적용) and one line per click in the pack's `physics_labels.jsonl`:
the object's features, what the rules said, what was set. That log is the material for the next review.

To review again: `python3.10 tools/physics_review.py --pack <pack> --scene <scene> --out <dir>`, judge, write
`<dir>/<scene>/verdicts.json`, then `python3 tools/physics_eval.py --review <dir> --pack <pack>`.
