# Examples

Two figures require real data and are skipped when it is absent, rather than
being replaced with illustrative substitutes.

## `fig_image_to_mesh`

Shows a source photograph beside front and isometric projections of the
generated and processed meshes. Requires:

```
examples/input/<photograph>.jpg     # or .png
examples/output/raw.stl             # backend output, before processing
examples/output/processed.stl       # after scaling and symmetry reconstruction
```

Produce them with:

```bash
python cli/generate.py examples/input/photo.jpg --size 30
cp "<generated path>" examples/output/raw.stl
python cli/symmetrize.py examples/output/raw.stl --axis x -o examples/output/processed.stl
```

## `fig_multiview`

Compares geometry reconstructed from one, two and four input views under
identical camera parameters, annotated with measured triangle count, volume and
component count. Requires:

```
examples/output/views_1.stl
examples/output/views_2.stl
examples/output/views_4.stl
```

Produce them from the same object, varying only the number of `--views`:

```bash
python cli/generate.py front.jpg --size 30
python cli/generate.py front.jpg --views back.jpg --size 30
python cli/generate.py front.jpg --views back.jpg left.jpg right.jpg --size 30
```

Then regenerate with `python figures/make_figures.py`.

Generated `.stl` files are excluded from version control by `.gitignore`.
