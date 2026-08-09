



-- Each line below puts a comment marker inside a string literal, with a
-- `${...}` after it on the same line. The marker is not a comment, so the
-- placeholder must still be templated.
select some_column, 'dashes -- inside a string' as lbl from `project.dataset.table_a`
where some_column != '/* not a comment */' and some_column = 'string_with_comment_marker'
