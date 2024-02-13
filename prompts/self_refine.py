HTML_CHECK_PROMPT = '''
Instruction: Given a question, we want an HTML that correctly shows the swaps and updated state after each swaps. Identify any discrepancies, misunderstandings, or errors in the Old HTML using the question as reference. If necessary, update the Old HTML based on the discrepancies, misunderstandings, or errors you identified. We want to focus on answering the question correctly using the image.


Q: Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Rodrigo, Bob is dancing with Jamie, and Claire is dancing with Lola.\nThroughout the song, the dancers often trade partners. First, Claire and Alice switch partners. Then, Bob and Claire switch partners. Finally, Claire and Alice switch partners. At the end of the dance, Alice is dancing with\nOptions:\n(A) Rodrigo\n(B) Jamie\n(C) Lola

# HTML Code:

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Square Dance Partners</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
        }}

        .container {{
            text-align: center;
            padding-top: 20px;
        }}

        .pair {{
            display: inline-block;
            margin: 10px;
        }}

        .dancer {{
            font-weight: bold;
        }}

        .partner {{
            border: 1px solid #ddd;
            margin-top: 5px;
            padding: 5px;
            background-color: #f9f9f9;
        }}

        .switch {{
            font-size: 20px;
            margin: 20px 0;
        }}
    </style>
</head>

<body>
    <div class="container">
        <h2>Starting Partners</h2>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Rodrigo</div>
        </div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Jamie</div>
        </div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Lola</div>
        </div>
        <h2>Switch 1: Claire ↔ Alice</h2>
        <div class="switch">↔️</div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Rodrigo</div>
        </div>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Lola</div>
        </div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Jamie</div>
        </div>
        <!-- Updated state after the swap: Claire has Rodrigo, Alice has Lola and Bob has Jamie. Use this state to update in later swaps.-->
        <h2>Switch 2: Bob ↔ Claire</h2>
        <div class="switch">↔️</div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Rodrigo</div>
        </div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Jamie</div>
        </div>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Lola</div>
        </div>
        <!-- Updated state after the swap: Bob has Rodrigo, Claire has Jamie and Alice has Lola. Use this state to update in later swaps.-->
        <h2>Switch 3: Claire ↔ Alice</h2>
        <div class="switch">↔️</div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Rodrigo</div>
        </div>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Jamie</div>
        </div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Lola</div>
        </div> <!-- Updated state after the swap: Claire has Rodrigo, Alice has Jamie and Bob has Lola.-->
    </div>
</body>

</html>


# There is an error in the HTML above because of lack of understanding of the question. What is the error? To find the error, go through the <div class="pair"> codes after each Switch, and check if everything looks consistent according to the question.

# Let us go through the error and check step-by-step
    <h2>Switch 1: Claire ↔ Alice</h2>
    <div class="switch">↔️</div>
    <div class="pair">
        <div class="dancer">Claire</div>
        <div class="partner">Rodrigo</div>
    </div>
    <div class="pair">
        <div class="dancer">Alice</div>
        <div class="partner">Lola</div>
    </div>
    <div class="pair">
        <div class="dancer">Bob</div>
        <div class="partner">Jamie</div>
    </div>
# looks good

# Let's check the other parts
    <h2>Switch 2: Bob ↔ Claire</h2>
    <div class="switch">↔️</div>
    <div class="pair">
        <div class="dancer">Bob</div>
        <div class="partner">Rodrigo</div>
    </div>
    <div class="pair">
        <div class="dancer">Claire</div>
        <div class="partner">Jamie</div>
    </div>
    <div class="pair">
        <div class="dancer">Alice</div>
        <div class="partner">Lola</div>
    </div>
# looks good

# Let's check the other parts
    <h2>Switch 3: Claire ↔ Alice</h2>
    <div class="switch">↔️</div>
    <div class="pair">
        <div class="dancer">Claire</div>
        <div class="partner">Rodrigo</div>
    </div>
    <div class="pair">
        <div class="dancer">Alice</div>
        <div class="partner">Jamie</div>
    </div>
    <div class="pair">
        <div class="dancer">Bob</div>
        <div class="partner">Lola</div>
    </div>
# wrong! There is a mistake here. Updated state after the last swap: Bob has Rodrigo, Claire has Jamie and Alice has Lola. In this swap between Claire and Alice, Claire should have Lola, not Rodrigo and Bob should have Rodrigo, not Lola.

Okay! Here is the updated HTML:

# Updated HTML:

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Square Dance Partners</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
        }}

        .container {{
            text-align: center;
            padding-top: 20px;
        }}

        .pair {{
            display: inline-block;
            margin: 10px;
        }}

        .dancer {{
            font-weight: bold;
        }}

        .partner {{
            border: 1px solid #ddd;
            margin-top: 5px;
            padding: 5px;
            background-color: #f9f9f9;
        }}

        .switch {{
            font-size: 20px;
            margin: 20px 0;
        }}
    </style>
</head>

<body>
    <div class="container">
        <h2>Starting Partners</h2>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Rodrigo</div>
        </div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Jamie</div>
        </div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Lola</div>
        </div>
        <h2>Switch 1: Claire ↔ Alice</h2>
        <div class="switch">↔️</div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Rodrigo</div>
        </div>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Lola</div>
        </div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Jamie</div>
        </div>
        <!-- Updated state after the swap: Claire has Rodrigo, Alice has Lola and Bob has Jamie. Use this state to update in later swaps.-->
        <h2>Switch 2: Bob ↔ Claire</h2>
        <div class="switch">↔️</div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Rodrigo</div>
        </div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Jamie</div>
        </div>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Lola</div>
        </div>
        <!-- Updated state after the swap: Bob has Rodrigo, Claire has Jamie and Alice has Lola. Use this state to update in later swaps.-->
        <h2>Switch 3: Claire ↔ Alice</h2>
        <div class="switch">↔️</div>
        <div class="pair">
            <div class="dancer">Claire</div>
            <div class="partner">Lola</div>
        </div>
        <div class="pair">
            <div class="dancer">Alice</div>
            <div class="partner">Jamie</div>
        </div>
        <div class="pair">
            <div class="dancer">Bob</div>
            <div class="partner">Rodrigo</div>
        </div> <!-- Updated state after the swap: Claire has Lola, Alice has Jamie and Bob has Rodrigo.-->
    </div>
</body>

</html>

### END ###

Q: {question}

#HTML Code:

{html}'''.strip()