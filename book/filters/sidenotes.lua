--[[
Footnotes as margin notes, Tufte-style: a numbered note beside the text on
a wide screen, and on a narrow one, behind a toggle on its number.
]]

local count = 0

function Note(el)
  count = count + 1
  local html = pandoc.write(pandoc.Pandoc(el.content), "html", {
    html_math_method = "katex",
  })
  -- a note sits inside a paragraph, so its paragraphs can't be <p>s
  html = html:gsub("<p>", '<span class="sn-p">'):gsub("</p>", "</span>"):gsub("\n", " ")
  return pandoc.RawInline("html", string.format(
    '<label for="sn-%d" class="margin-toggle sidenote-number"><sup>%d</sup></label>'
      .. '<input type="checkbox" id="sn-%d" class="margin-toggle"/>'
      .. '<span class="sidenote"><sup>%d</sup> %s</span>',
    count, count, count, count, html
  ))
end
