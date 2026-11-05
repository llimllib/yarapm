--[[
Numbering and cross-references for the book (book/build.py).

Metadata, set by the build:
  chapter       the chapter's number
  file          the chapter's output file, such as 04-ridge.html
  collect       pass 1: write this chapter's labels and sections here (JSON)
  labels        pass 2: every chapter's labels (JSON), to resolve references
  code-commit   the commit "code:" links point at

Numbered per chapter, as "Definition 3.2":
  ## and ### headers               Section 3.2, 3.2.1 (not .unnumbered)
  ::: definition, theorem, proposition, lemma, corollary, example, remark
                                   one shared counter
  ::: exercise                     its own counter
  ::: code                         "In the code", unnumbered
  ::: proof                        unnumbered, ends with a QED mark
  $$ ... $$ {#eq:name}             (3.1), with KaTeX's \tag
  ![caption](figures/x.svg){#fig:name}   the SVG inlined, "Figure 3.1."
  a table whose caption ends in {#tbl:name}   "Table 3.1."

An environment's title comes from its `title` attribute. A reference is a
citation of a label: @def:ridge is "Definition 4.2", [-@def:ridge] just
"4.2", @eq:normal "(3.4)". A link to code:yarapm/rapm.py#L124 goes to that
file on GitHub at code-commit.
]]

local stringify = pandoc.utils.stringify

local REPO = "https://github.com/llimllib/yarapm/blob/"

local ENVIRONMENTS = {
  definition = { name = "Definition", counter = "env" },
  theorem = { name = "Theorem", counter = "env" },
  proposition = { name = "Proposition", counter = "env" },
  lemma = { name = "Lemma", counter = "env" },
  corollary = { name = "Corollary", counter = "env" },
  example = { name = "Example", counter = "env" },
  remark = { name = "Remark", counter = "env" },
  exercise = { name = "Exercise", counter = "exercise" },
  code = { name = "In the code" },
  proof = { name = "Proof" },
}

local chapter = "0"
local file = ""
local collect = nil
local commit = "main"
local labels = {}
local counters = { env = 0, exercise = 0, figure = 0, equation = 0, table = 0 }
local section, subsection = 0, 0
local sections = {}

local function register(id, name, number)
  if id == nil or id == "" then
    return
  end
  labels[id] = { name = name, number = number, file = file }
end

local function next_number(counter)
  counters[counter] = counters[counter] + 1
  return chapter .. "." .. counters[counter]
end

local function read_meta(meta)
  chapter = meta.chapter and stringify(meta.chapter) or "0"
  file = meta.file and stringify(meta.file) or ""
  commit = meta["code-commit"] and stringify(meta["code-commit"]) or "main"
  if meta.collect then
    collect = stringify(meta.collect)
  end
  if meta.labels then
    local f = assert(io.open(stringify(meta.labels)))
    labels = pandoc.json.decode(f:read("a"), false)
    f:close()
  end
  if meta.label then
    register(stringify(meta.label), "Chapter", chapter)
  end
end

local function Header(el)
  if el.classes:includes("unnumbered") or el.level > 3 then
    return el
  end
  local number
  if el.level == 2 then
    section = section + 1
    subsection = 0
    number = chapter .. "." .. section
    table.insert(sections, {
      number = number,
      title = stringify(el.content),
      id = el.identifier,
    })
  elseif el.level == 3 then
    subsection = subsection + 1
    number = chapter .. "." .. section .. "." .. subsection
  else
    return el
  end
  register(el.identifier, "Section", number)
  el.content:insert(1, pandoc.Space())
  el.content:insert(1, pandoc.Span(number, { class = "num" }))
  return el
end

local function Div(el)
  local kind = nil
  for _, class in ipairs(el.classes) do
    if ENVIRONMENTS[class] then
      kind = class
      break
    end
  end
  if kind == nil then
    return el
  end
  local env = ENVIRONMENTS[kind]
  local label = env.name
  if env.counter then
    local number = next_number(env.counter)
    label = label .. " " .. number
    register(el.identifier, env.name, number)
  end
  local head = pandoc.Inlines({ pandoc.Span(label, { class = "label" }) })
  local title = el.attributes.title
  if title then
    head:insert(pandoc.Space())
    head:insert(pandoc.Span(pandoc.read("(" .. title .. ")", "markdown").blocks[1].content,
      { class = "name" }))
    el.attributes.title = nil
  end
  head:insert(pandoc.Str("."))
  head:insert(pandoc.Space())
  local first = el.content[1]
  if first and (first.t == "Para" or first.t == "Plain") then
    first.content = head .. first.content
  else
    el.content:insert(1, pandoc.Para(head))
  end
  if kind == "proof" then
    local last = el.content[#el.content]
    if last.t == "Para" or last.t == "Plain" then
      last.content:insert(pandoc.Span("∎", { class = "qed" }))
    end
  end
  el.classes = pandoc.List({ "env", kind })
  return el
end

-- display math followed by {#eq:name}: number it
local function number_equations(inlines)
  local out = pandoc.Inlines({})
  local i = 1
  while i <= #inlines do
    local el = inlines[i]
    if el.t == "Math" and el.mathtype == "DisplayMath" then
      local j = i + 1
      if inlines[j] and inlines[j].t == "Space" then
        j = j + 1
      end
      local tag = inlines[j]
      local id = tag and tag.t == "Str" and tag.text:match("^{#(eq:[^}]+)}$")
      if id then
        local number = next_number("equation")
        register(id, "Equation", number)
        el.text = el.text .. " \\tag{" .. number .. "}"
        out:insert(pandoc.Span({ el }, { id = id, class = "equation" }))
        i = j + 1
      else
        out:insert(el)
        i = i + 1
      end
    else
      out:insert(el)
      i = i + 1
    end
  end
  return out
end

local function Para(el)
  el.content = number_equations(el.content)
  return el
end

local function Figure(el)
  local image = nil
  pandoc.walk_block(el, {
    Image = function(img)
      image = img
    end,
  })
  local number = next_number("figure")
  register(el.identifier, "Figure", number)
  if image and image.src:match("%.svg$") then
    local f = assert(io.open(image.src), "missing figure " .. image.src)
    local svg = f:read("a"):gsub("^<%?xml.-%?>%s*", "")
    f:close()
    el.content = { pandoc.RawBlock("html", svg) }
  end
  local caption = el.caption.long
  local label = pandoc.Span("Figure " .. number .. ".", { class = "label" })
  if #caption > 0 and (caption[1].t == "Plain" or caption[1].t == "Para") then
    caption[1].content = pandoc.Inlines({ label, pandoc.Space() }) .. caption[1].content
  else
    caption:insert(1, pandoc.Plain({ label }))
  end
  el.caption.long = caption
  return el
end

local function Table(el)
  local caption = el.caption.long
  if #caption == 0 then
    return el
  end
  local block = caption[#caption]
  local content = block.content
  local id = nil
  if content and #content > 0 and content[#content].t == "Str" then
    id = content[#content].text:match("^{#(tbl:[^}]+)}$")
    if id then
      content:remove(#content)
      if #content > 0 and content[#content].t == "Space" then
        content:remove(#content)
      end
    end
  end
  -- pandoc reads a caption's {#id} itself, in newer versions
  id = id or el.identifier
  local number = next_number("table")
  register(id, "Table", number)
  el.identifier = id
  local first = caption[1]
  first.content = pandoc.Inlines({
    pandoc.Span("Table " .. number .. ".", { class = "label" }),
    pandoc.Space(),
  }) .. first.content
  return el
end

local function reference(id, suppress)
  local target = labels[id]
  if target == nil then
    return nil
  end
  local text
  if target.name == "Equation" then
    text = "(" .. target.number .. ")"
  elseif suppress then
    text = target.number
  else
    text = target.name .. "\u{a0}" .. target.number
  end
  local href = (target.file == file and "" or target.file) .. "#" .. id
  if target.name == "Chapter" then
    href = target.file
  end
  return pandoc.Link(text, href, "", { class = "ref" })
end

local function Cite(el)
  if collect then
    return el
  end
  local out = pandoc.Inlines({})
  for i, citation in ipairs(el.citations) do
    local link = reference(citation.id, citation.mode == "SuppressAuthor")
    if link == nil then
      error("unknown reference @" .. citation.id .. " in " .. file)
    end
    if i > 1 then
      out:insert(pandoc.Str(","))
      out:insert(pandoc.Space())
    end
    out:extend(citation.prefix)
    if #citation.prefix > 0 then
      out:insert(pandoc.Space())
    end
    out:insert(link)
  end
  return out
end

local function Link(el)
  local path = el.target:match("^code:(.+)$")
  if path then
    el.target = REPO .. commit .. "/" .. path
    el.classes:insert("src")
  end
  return el
end

local function finish(doc)
  if collect then
    local mine = {}
    for id, label in pairs(labels) do
      if label.file == file then
        mine[id] = label
      end
    end
    local f = assert(io.open(collect, "w"))
    f:write(pandoc.json.encode({
      labels = mine,
      sections = sections,
      title = stringify(doc.meta.title or ""),
    }))
    f:close()
  end
  return doc
end

return {
  { Meta = read_meta },
  {
    traverse = "topdown",
    Header = Header,
    Div = Div,
    Para = Para,
    Plain = Para,
    Figure = Figure,
    Table = Table,
  },
  { Cite = Cite, Link = Link },
  { Pandoc = finish },
}
