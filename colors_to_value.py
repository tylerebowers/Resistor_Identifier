

DIGIT = {
    'black': 0, 'brown': 1, 'red': 2, 'orange': 3, 'yellow': 4,
    'green': 5, 'blue': 6, 'violet': 7, 'grey': 8, 'white': 9
}

MULTIPLIER = {
    'black': 1,
    'brown': 10,
    'red': 100,
    'orange': 1e3,
    'yellow': 1e4,
    'green': 1e5,
    'blue': 1e6,
    'violet': 1e7,
    'grey': 1e8,
    'white': 1e9,
    'gold': 0.1,
    'silver': 0.01
}

TOLERANCE = {
    'brown': 1,
    'red': 2,
    'green': 0.5,
    'blue': 0.25,
    'violet': 0.1,
    'grey': 0.05,
    'gold': 5,
    'silver': 10,
    # if no band is present, ±20%
    'none': 20
}

#ppm/K
TEMPCOEF = {
    'brown': 250,
    'brown': 100,
    'red': 50,
    'orange': 15,
    'yellow': 25,
    'blue': 10,
    'violet': 5,
    'grey': 1
}

def parse_resistor(colors, unicode=True):
    def _format_value(ohms):
        """Format ohm value into a readable string with suffix."""
        if ohms >= 1e6:
            return f"{ohms/1e6:.2f}M"
        elif ohms >= 1e3:
            return f"{ohms/1e3:.2f}k"
        else:
            return f"{ohms:.2f}"

    n = len(colors)

    if n == 0:
        return "No bands detected."
    else:
        for c in colors:
            if c not in MULTIPLIER:
                return "Invalid color detected."


    if n > 2:
        if colors[0] in ['gold', 'silver'] or colors[1] in ['gold', 'silver']: 
            colors = colors[::-1]
        
    if n == 1:
        return 0
    elif n == 3:
        d1 = DIGIT.get(colors[0], None)
        d2 = DIGIT.get(colors[1], None)
        mult = MULTIPLIER.get(colors[2], None)
        if None in (d1,d2,mult): return "?"
        base = (d1*10 + d2) * mult
        return f"{_format_value(base)} {'Ω ±' if unicode else 'Ohms +/-'}20%"
    elif n == 4:
        d1 = DIGIT.get(colors[0], None)
        d2 = DIGIT.get(colors[1], None)
        mult = MULTIPLIER.get(colors[2], None)
        if None in (d1,d2,mult): return "?"
        base = (d1*10 + d2) * mult
        tol = TOLERANCE.get(colors[3], "?")
        return f"{_format_value(base)} {'Ω ±' if unicode else 'Ohms +/-'}{tol}%"
    elif n == 5:
        d1 = DIGIT.get(colors[0], None)
        d2 = DIGIT.get(colors[1], None)
        d3 = DIGIT.get(colors[2], None)
        mult = MULTIPLIER.get(colors[3], None)
        if None in (d1,d2,d3,mult): return "?"
        base = (d1*100 + d2*10 + d3) * mult
        tol = TOLERANCE.get(colors[4], "?")
        return f"{_format_value(base)} {'Ω ±' if unicode else 'Ohms +/-'}{tol}%"
    elif n == 6:
        d1 = DIGIT.get(colors[0], None)
        d2 = DIGIT.get(colors[1], None)
        d3 = DIGIT.get(colors[2], None)
        mult = MULTIPLIER.get(colors[3], None)
        if None in (d1,d2,d3,mult): return "?"
        base = (d1*100 + d2*10 + d3) * mult
        tol = TOLERANCE.get(colors[4], "?")
        tc = TEMPCOEF.get(colors[5], "?")
        tc_str = f", {tc} ppm/K" if tc is not None else ""
        return f"{_format_value(base)} {'Ω ±' if unicode else 'Ohms +/-'}{tol}%{tc_str}"
    else:
        return "Out of Range."


if __name__ == "__main__":
    examples = [["red", "black", "orange", "blue"],
                ["brown", "green", "black", "red", "gold"],
                ["orange", "orange", "black", "brown", "brown", "red"]]

    for example in examples:    
        print(example, "=>", parse_resistor(example))

