# Evil Express

This is an extendable parser for mathematical expressions.

In Evil-Express-land, unary operators always have higher priority than binary operators, e.g., 
`-2^2=4` instead of `-2^2=-4`. Moreover, we are not too strict regarding parantheses. 
For instance `"---1"` will evalute to `-1`. 
If you want to be on the safe side, I suggest using parantheses.