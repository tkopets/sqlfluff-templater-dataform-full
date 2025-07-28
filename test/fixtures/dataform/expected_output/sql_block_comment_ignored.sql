

/* the block below should be ignored */

/*
    select * from ${ ref('NO_SUCH_TABLE') }
*/

with regular as (
    select * from `project.dataset.table_a`
)
select *
from regular
where regular.some_column = 'line_comment_test'
