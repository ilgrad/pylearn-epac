Build pages
-----------

We assume that your epac repository is located in "~/pylearn-epac". Goto the directory which contains the doc source, and then build the doc.

```
$ EPACDIR=~/pylearn-epac
$ cd $EPACDIR/doc/source
$ make html
$ mkdir -p ./_build/html/epydoc_api
$ epydoc -v --html epac -o ./_build/html/epydoc_api
```

The website has been built in "~/pylearn-epac/doc/source/_build/html". You can open "index.html" by firefox to test.

```
$ firefox  $EPACDIR/doc/source/_build/html/index.html
```

Upload to github
----------------
"$EPACDIR/doc/source/_build/html" contains the epac website. Now we start to upload to github server. Clone epac from github to a temporary directory, and checkout gh-pages branch

```
$ cd /tmp
$ git clone git@github.com:neurospin/pylearn-epac.git epac_doc
$ cd epac_doc
$ git fetch origin
$ git checkout -b gh-pages origin/gh-pages
```

Copy the built website and push to the gh-pages branch on github.

```
$ cp -r $EPACDIR/doc/source/_build/html/* ./
$ git add .
$ git commit -a -m "DOC: update pages"
$ git push origin gh-pages
```

Now, you can visit your updated website at http://neurospin.github.io/pylearn-epac.