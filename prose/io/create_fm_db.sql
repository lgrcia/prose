CREATE TABLE files (
    date text, 
    path text PRIMARY KEY,
    telescope text, 
    filter text, 
    type text, 
    target text, 
    width int, 
    height int, 
    jd real, 
    id int, 
    exposure real,
    FOREIGN KEY(id) REFERENCES observations(id)
);

CREATE TABLE observations (
  id INTEGER PRIMARY KEY,
  date text,
  telescope text,
  filter text,
  type text,
  target text,
  width int,
  height int,
  exposure real,
  files int,
  UNIQUE(date, telescope, filter, target, type, width, height, exposure)
);

CREATE TABLE products (
  id int,
  datetime text,
  version text,
  files int,
  path text,
  status text,
  FOREIGN KEY(id) REFERENCES observations(id)
);