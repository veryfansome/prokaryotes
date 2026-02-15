CREATE CONSTRAINT FOR (f:Fact)            REQUIRE f.doc_id  IS UNIQUE;

CREATE CONSTRAINT FOR (n:NamedEntity)     REQUIRE n.text    IS UNIQUE;

CREATE CONSTRAINT FOR (p:Prompt)          REQUIRE p.doc_id  IS UNIQUE;

CREATE CONSTRAINT FOR (r:Response)        REQUIRE r.doc_id  IS UNIQUE;

CREATE CONSTRAINT FOR (t:ToolCall)        REQUIRE t.doc_id  IS UNIQUE;

CREATE CONSTRAINT FOR (t:Topic)           REQUIRE t.text    IS UNIQUE;
