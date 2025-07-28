['b', 'c'].forEach((suffix) => {
    declare({
        database: "project",
        schema: "dataset",
        name: "table_" + suffix,
    });
});
