main = do
    text <- readFile "/guibrunhole/unicamp/mc346/haskell/graph.txt"
    let cases = lines text
        -- to get rid of the periods at the end of each line
        strs = map init cases
        lastLine = read $ last strs
    print $ show (map (+5) lastLine)
