-- HOW TO EXECUTE

:load dijkstra.hs

txt <- readFile "graph.txt"

let lines_file = lines txt
let target = head (reverse lines_file)
let new_lines_file = init lines_file
let origin = head (reverse new_lines_file)

let new_txt = init(init(init(init txt)))
let g = fromText new_txt False
let soln = dijkstra g origin -- ponto inicial
let path = pathToNode soln target
let cost = map fst (map snd soln)!!(position target (map fst soln)) -- ponto final

putStrLn ("Inicial: "++origin)
putStrLn ("Final: "++target)
putStrLn ("Caminho: ")
path
putStrLn ("Custo: ")
cost


