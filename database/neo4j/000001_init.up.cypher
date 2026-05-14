CREATE CONSTRAINT FOR (n:NamedEntity)     REQUIRE n.text    IS UNIQUE;

CREATE CONSTRAINT FOR (t:Topic)           REQUIRE t.text    IS UNIQUE;
