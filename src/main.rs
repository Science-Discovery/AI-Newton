struct Foo {
    a: i32,
}
impl std::fmt::Display for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(width) = f.width() {
            write!(f, "{:width$}--", self.a, width = width)
        } else {
            write!(f, "{}--", self.a)
        }
    }
}

fn main() {
    println!("Hello, world!");
    // println!("-{: ^20}-", "Hello, world!");
    // // 开头输出10个空格，再输出Hello, world!
    // println!("{:10}{}", " ", "Hello, world!");
    // // 输出4个空格，再输出Hello, world!
    // println!("{}{}", format_args!("{: >1$}", "", 4), "Hello world!");
    // println!("{:4}Indented text!", "");
    println!("{}", format!("{:10}", Foo { a: 42 }));
}
