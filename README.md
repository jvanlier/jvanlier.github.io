jvanlier.github.io
==================

Used for freelancing activities.


## Setup local env

Make sure we use a brew env rather than system ruby in order to not use root to install dependencies.

```bash
eval "$(rbenv init -)"
rbenv global 2.6.9
```

May want to use `local` if if Ruby is used for anything else, or a virtualenv-equivalent thing. I only use Ruby for this so for me it's fine.

Install deps:

```bash
bundle install
```

Run local server:

```bash
bundle exec jekyll serve
```
