grammar semantic;

file: semantics ;

semantics: (WHITESPACE* symbol WHITESPACE)*  ;

symbol: type (DASH identifier)? ;

type: STRING ;
identifier: (STRING | '_')+;

COMMENT : '%' ~[\r\n]* -> skip;
STRING : [a-zA-Z0-9.#/]+;
DASH: '-' ;
WHITESPACE : [ \r\n\t]+ ;
